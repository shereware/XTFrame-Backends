#!/usr/bin/env python3
"""T-671 (Docs/design/50-c-abi.md §50.ab addendum 2, item 4): run xt_gpu_mem_probe over a matrix of wgpu memory
strategies and summarize. Standard library only; works from Git Bash and PowerShell alike:

    python Tools/Scripts/gpu_mem_matrix.py [--config FILE] [--bundle DIR] [--out DIR] [--timeout S]
                                           [--only SUBSTRING] [--dry-run] [--compare OLD_DIR [--against NEW_DIR]]
    python Tools/Scripts/gpu_mem_matrix.py --self-test

How to build the trunk wgpu-native and the probes (Windows; the main checkout or a worktree):
    bash Tools/Scripts/build_wgpu_native_trunk.sh b83eb64        # -> build/wgpu-native-trunk/current
    cmake --preset vs2022-wgpu-trunk
    cmake --build --preset vs2022-wgpu-trunk-release --target xt_gpu_mem_probe   # the "trunk" build
    cmake --preset vs2022 -DXT_BUILD_CAPI=ON
    cmake --build --preset vs2022-release --target xt_gpu_mem_probe              # the "pinned" build (v29.0.1.1)

How to run: `python Tools/Scripts/gpu_mem_matrix.py` runs every combination of the default config
(Tools/Scripts/gpu_mem_matrix.default.json: memory hints x msaa x backend x power x trim per build) one probe
process at a time, each with XT_WGPU_MEMORY_HINTS set and TMP/TEMP pointing into the output directory, killed
after `timeout_s` (default 300 s). A failed or timed-out combination is marked in the report and the others
still run. Output: build/gpu-mem/<timestamp>/ with report.md, summary.json and per combination <id>.csv (the
probe's rows) and <id>.log (its stdout/stderr: adapter, wgpu-native version, memory hints in effect).

How to compare two runs: `python Tools/Scripts/gpu_mem_matrix.py --compare build/gpu-mem/<old> --against
build/gpu-mem/<new>` prints (and writes into <new>/compare.md) the difference of every metric for the
combinations both runs have; `--compare <old>` without --against compares the run it has just made.

`--dry-run` lists the combinations and commands without running anything; `--self-test` checks the matrix
expansion, the CSV summary and the comparison on built-in data.

T-683 (50 §50.ad): the config keys `prewarm_hidden` (N > 0 adds `--prewarm-hidden N`: every surface is created
hidden and updated for N frames before the first show, the host flow) and `system_fonts` (false adds
`--no-system-fonts`) are passed to every probe; absent = 0 / true, the probe's own defaults.
"""

import argparse
import csv
import datetime
import io
import itertools
import json
import os
import subprocess
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_CONFIG = os.path.join(ROOT, "Tools", "Scripts", "gpu_mem_matrix.default.json")
MB = 1024.0 * 1024.0

# Metrics of summary.json, in report column order: (key, label).
METRICS = [
    ("create_private", "create private"), ("create_local", "create local"), ("create_nonlocal", "create non-local"),
    ("first_private", "1st scene private"), ("first_local", "1st scene local"), ("first_nonlocal", "1st scene non-local"),
    ("peak_private", "peak private"), ("peak_local", "peak local"), ("peak_nonlocal", "peak non-local"),
    ("end_private", "end private"), ("end_local", "end local"), ("end_nonlocal", "end non-local"),
    ("max_step_private", "max step private"), ("max_first_ms", "max 1st-frame ms"),
]


def expand(config):
    """The combinations of config["matrix"]: a list of dicts (build, hints, msaa, backend, power, trim, id)."""
    combos = []
    for entry in config["matrix"]:
        dims = itertools.product(entry["memory_hints"], entry["msaa"], entry["backend"],
                                 entry.get("power", ["high"]), entry.get("trim", [0]))
        for hints, msaa, backend, power, trim in dims:
            combo = {"build": entry["build"], "hints": hints, "msaa": int(msaa), "backend": backend,
                     "power": power, "trim": int(trim)}
            combo["id"] = combo_id(combo)
            combos.append(combo)
    return combos


def combo_id(c):
    return "%s_%s_msaa%d_%s_%s_trim%d" % (c["build"], c["hints"].replace(":", "-"), c["msaa"], c["backend"] or "default",
                                         c["power"], c["trim"])


def resolve(path):
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(ROOT, path))


def probe_command(config, combo, csv_path):
    return [resolve(config["builds"][combo["build"]]), "--bundle", config["bundle"], "--out", csv_path,
            "--sequence", config["sequence"], "--rounds", str(config["rounds"]),
            "--frames-per-scene", str(config["frames_per_scene"]), "--width", str(config["width"]),
            "--height", str(config["height"]), "--msaa", str(combo["msaa"]), "--backend", combo["backend"],
            "--power", combo["power"], "--mode", config["mode"], "--trim-each-scene", str(combo["trim"]),
            "--image-budget", str(config["image_budget"])] + host_flow_args(config)


def host_flow_args(config):
    """T-683: the host-flow options of the config (absent: none)."""
    args = []
    if int(config.get("prewarm_hidden", 0)) > 0:
        args += ["--prewarm-hidden", str(int(config["prewarm_hidden"]))]
    if not config.get("system_fonts", True):
        args.append("--no-system-fonts")
    return args


def num(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def summarize(csv_text):
    """summary.json metrics (MB / ms) of one probe CSV; None where the CSV has no value."""
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    out = {k: None for k, _ in METRICS}
    create = next((r for r in rows if r["phase"] == "target_create"), None)
    scenes = [r for r in rows if r["phase"] == "scene"]
    mb = lambda r, k: (num(r, k) / MB) if r is not None and num(r, k) is not None else None
    for prefix, row in (("create", create), ("first", scenes[0] if scenes else None),
                        ("end", scenes[-1] if scenes else None)):
        out[prefix + "_private"] = mb(row, "private_bytes")
        out[prefix + "_local"] = mb(row, "dxgi_local_bytes")
        out[prefix + "_nonlocal"] = mb(row, "dxgi_nonlocal_bytes")
    live = ([create] if create else []) + scenes + [r for r in rows if r["phase"] == "teardown"]
    for key, col in (("peak_private", "private_bytes"), ("peak_local", "dxgi_local_bytes"),
                     ("peak_nonlocal", "dxgi_nonlocal_bytes")):
        values = [mb(r, col) for r in live if mb(r, col) is not None]
        out[key] = max(values) if values else None
    steps = [mb(b, "private_bytes") - mb(a, "private_bytes") for a, b in zip(live, live[1:])
             if mb(a, "private_bytes") is not None and mb(b, "private_bytes") is not None]
    out["max_step_private"] = max(steps) if steps else None
    firsts = [num(r, "first_frame_ms") for r in scenes if num(r, "first_frame_ms") is not None]
    out["max_first_ms"] = max(firsts) if firsts else None
    out["scenes"] = len(scenes)
    return out


def adapter_of(log_text):
    """The adapter and memory-hints lines the library logged at target creation."""
    adapter, memory = "", ""
    for line in log_text.splitlines():
        if "xt_gpu_target: adapter " in line and not adapter:
            adapter = line.split("xt_gpu_target: adapter ", 1)[1].strip()
        elif "xt_gpu_target: wgpu-native " in line and not memory:
            memory = line.split("xt_gpu_target: ", 1)[1].strip()
    return adapter, memory


def fmt(v, digits=1):
    return "" if v is None else ("%.*f" % (digits, v))


def triple(s, prefix):
    return "%s / %s / %s" % (fmt(s.get(prefix + "_private")), fmt(s.get(prefix + "_local")),
                             fmt(s.get(prefix + "_nonlocal")))


def report_markdown(config, results, header_lines):
    lines = ["# GPU memory matrix", ""] + header_lines + ["",
             "Columns: MB as private / DXGI local / DXGI non-local of the adapter in use. *create* = after "
             "`xt_gpu_target_create`; *1st scene* = after the first UI's frames; *peak* = maximum over create, every "
             "scene row and teardown; *end* = after the last scene of the last round; *max step* = largest private "
             "growth between consecutive sample points; *1st-frame ms* = maximum over the scenes of the first frame "
             "after a UI is shown (CPU wall time of begin_frame..end_frame).", "",
             "| combination | status | adapter | create | 1st scene | peak | end | max step | max 1st-frame ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = r.get("summary") or {}
        adapter = r.get("adapter", "")
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["id"], r["status"], adapter.split(", power")[0] if adapter else "", triple(s, "create"),
            triple(s, "first"), triple(s, "peak"), triple(s, "end"), fmt(s.get("max_step_private")),
            fmt(s.get("max_first_ms"))))
    lines += ["", "## Strategy lines logged at target creation", ""]
    for r in results:
        lines.append("- `%s`: %s; %s" % (r["id"], r.get("adapter", "") or "(no adapter line)",
                                         r.get("memory", "") or "(no memory line)"))
    return "\n".join(lines) + "\n"


def compare(old, new):
    """Markdown table of new - old for every metric of the combinations in both summary dicts."""
    keys = [k for k, _ in METRICS]
    lines = ["| combination | " + " | ".join(label for _, label in METRICS) + " |",
             "|---|" + "---|" * len(keys)]
    for cid in sorted(set(old) & set(new)):
        cells = []
        for k in keys:
            a, b = (old[cid].get("summary") or {}).get(k), (new[cid].get("summary") or {}).get(k)
            cells.append("" if a is None or b is None else "%+.1f" % (b - a))
        lines.append("| %s | %s |" % (cid, " | ".join(cells)))
    only = sorted(set(old) ^ set(new))
    if only:
        lines += ["", "Only in one run: " + ", ".join(only)]
    return "\n".join(lines) + "\n"


def load_summary(directory):
    with open(os.path.join(directory, "summary.json"), encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)["results"]}


def git_head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def run_matrix(config, combos, out_dir, timeout):
    tmp = os.path.join(out_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    results = []
    for n, combo in enumerate(combos, 1):
        csv_path = os.path.join(out_dir, combo["id"] + ".csv")
        log_path = os.path.join(out_dir, combo["id"] + ".log")
        env = dict(os.environ, XT_WGPU_MEMORY_HINTS=combo["hints"], TMP=tmp, TEMP=tmp)
        cmd = probe_command(config, combo, csv_path)
        print("[%d/%d] %s" % (n, len(combos), combo["id"]), flush=True)
        result = dict(combo, status="ok")
        started = time.time()
        if not os.path.isfile(cmd[0]):
            result["status"] = "no probe (%s)" % cmd[0]
            log = ""
        else:
            try:
                p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout,
                                   encoding="utf-8", errors="replace")
                log = p.stdout + p.stderr
                if p.returncode != 0:
                    result["status"] = "exit %d" % p.returncode
                elif "SKIP: no WebGPU adapter" in p.stdout:
                    result["status"] = "skip (no adapter)"
            except subprocess.TimeoutExpired as e:
                partial = e.stdout or ""
                log = partial if isinstance(partial, str) else partial.decode("utf-8", "replace")
                result["status"] = "TIMEOUT (%d s, killed)" % timeout
        result["seconds"] = round(time.time() - started, 1)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log)
        result["adapter"], result["memory"] = adapter_of(log)
        if os.path.isfile(csv_path):
            with open(csv_path, encoding="utf-8") as f:
                result["summary"] = summarize(f.read())
        results.append(result)
        print("    %s, %.1f s, %s" % (result["status"], result["seconds"], result["adapter"]), flush=True)
    return results


SELF_TEST_CSV = """round,scene,phase,ui,private_bytes,working_set_bytes,dxgi_local_bytes,dxgi_nonlocal_bytes,image_textures,image_bytes,image_idle_textures,image_idle_bytes,image_evictions,attached_surfaces,surface_bytes,transient_bytes,atlas_bytes,first_frame_ms,steady_frame_ms,elapsed_ms
0,0,start,,1048576,1048576,,,,,,,,,,,,,,0.1
0,0,target_create,,104857600,1,52428800,10485760,0,0,0,0,0,0,0,0,0,,,10
1,1,scene,a,209715200,1,62914560,10485760,0,0,0,0,0,1,0,0,0,20.5,1.0,20
1,2,scene,b,157286400,1,73400320,10485760,0,0,0,0,0,2,0,0,0,30.25,1.0,30
0,0,teardown,,157286400,1,73400320,10485760,0,0,0,0,0,0,0,0,0,,,40
0,0,end,,1048576,1,0,0,,,,,,,,,,,,50
"""


def self_test():
    absolute = os.path.join(os.path.abspath(os.sep), "abs", "probe.exe")
    config = {"matrix": [{"build": "t", "memory_hints": ["usage", "manual:16:64"], "msaa": [4, 1],
                          "backend": ["d3d12"], "power": ["high", "low"], "trim": [0]},
                         {"build": "p", "memory_hints": ["performance"], "msaa": [4], "backend": ["d3d12"]}],
              "builds": {"t": "probe.exe", "p": absolute}, "bundle": "B", "sequence": "all", "rounds": 1,
              "frames_per_scene": 2, "width": 8, "height": 8, "mode": "hide", "image_budget": "default"}
    combos = expand(config)
    assert len(combos) == 2 * 2 * 2 + 1, len(combos)
    assert combos[0]["id"] == "t_usage_msaa4_d3d12_high_trim0", combos[0]["id"]
    assert combos[2]["id"] == "t_usage_msaa1_d3d12_high_trim0", combos[2]["id"]
    assert any(c["id"] == "t_manual-16-64_msaa1_d3d12_low_trim0" for c in combos)
    assert combos[-1]["power"] == "high" and combos[-1]["trim"] == 0
    cmd = probe_command(config, combos[-1], "x.csv")
    assert cmd[0] == absolute and cmd[cmd.index("--msaa") + 1] == "4", cmd
    assert cmd[cmd.index("--power") + 1] == "high" and cmd[cmd.index("--trim-each-scene") + 1] == "0"
    assert "--prewarm-hidden" not in cmd and "--no-system-fonts" not in cmd, cmd
    host = probe_command(dict(config, prewarm_hidden=120, system_fonts=False), combos[-1], "x.csv")
    assert host[host.index("--prewarm-hidden") + 1] == "120" and host[-1] == "--no-system-fonts", host
    s = summarize(SELF_TEST_CSV)
    assert s["create_private"] == 100.0 and s["create_local"] == 50.0 and s["create_nonlocal"] == 10.0, s
    assert s["first_private"] == 200.0 and s["end_private"] == 150.0 and s["end_local"] == 70.0, s
    assert s["peak_private"] == 200.0 and s["peak_local"] == 70.0, s
    assert s["max_step_private"] == 100.0, s
    assert s["max_first_ms"] == 30.25 and s["scenes"] == 2, s
    adapter, memory = adapter_of("log0 xt_gpu_target: adapter 'X', d3d12, integrated, power low\n"
                                 "log0 xt_gpu_target: wgpu-native 0.0.0.0 (override), memory hints usage\n")
    assert adapter == "'X', d3d12, integrated, power low" and memory.endswith("memory hints usage"), (adapter, memory)
    old = {"a": {"summary": dict(s)}, "gone": {"summary": dict(s)}}
    newer = dict(s, peak_private=180.0)
    table = compare(old, {"a": {"summary": newer}, "new": {"summary": s}})
    assert "| a | +0.0 |" in table and "-20.0" in table and "Only in one run: gone, new" in table, table
    md = report_markdown(config, [dict(combos[0], status="ok", summary=s, adapter=adapter, memory=memory),
                                  dict(combos[1], status="TIMEOUT (300 s, killed)")], ["header"])
    assert "| t_usage_msaa4_d3d12_high_trim0 | ok | 'X', d3d12, integrated |" in md, md
    assert "TIMEOUT (300 s, killed)" in md
    print("gpu_mem_matrix self-test: OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--bundle", help="override the config's bundle directory")
    ap.add_argument("--out", help="output directory (default build/gpu-mem/<timestamp>)")
    ap.add_argument("--timeout", type=int, help="seconds per probe process (default: config timeout_s or 300)")
    ap.add_argument("--only", help="run only the combinations whose id contains this text")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compare", metavar="OLD_DIR")
    ap.add_argument("--against", metavar="NEW_DIR", help="with --compare: compare to this run instead of running")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.compare and args.against:
        table = compare(load_summary(args.compare), load_summary(args.against))
        with open(os.path.join(args.against, "compare.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("# %s vs %s\n\n%s" % (args.against, args.compare, table))
        print(table)
        return 0
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    if args.bundle:
        config["bundle"] = args.bundle
    timeout = args.timeout or int(config.get("timeout_s", 300))
    combos = [c for c in expand(config) if not args.only or args.only in c["id"]]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.abspath(args.out or os.path.join(ROOT, "build", "gpu-mem", stamp))
    if args.dry_run:
        for c in combos:
            print("%s\n    XT_WGPU_MEMORY_HINTS=%s %s" % (c["id"], c["hints"], " ".join(
                probe_command(config, c, os.path.join(out_dir, c["id"] + ".csv")))))
        print("%d combination(s); output would go to %s" % (len(combos), out_dir))
        return 0
    os.makedirs(out_dir, exist_ok=True)
    results = run_matrix(config, combos, out_dir, timeout)
    header = ["- date: %s" % datetime.datetime.now().isoformat(timespec="seconds"),
              "- repository HEAD: %s" % git_head(),
              "- bundle: `%s`, sequence `%s`, %s rounds x %s frames, %sx%s, mode %s, image budget %s, timeout %d s" % (
                  config["bundle"], config["sequence"], config["rounds"], config["frames_per_scene"], config["width"],
                  config["height"], config["mode"], config["image_budget"], timeout),
              "- builds: " + ", ".join("%s = `%s`" % (k, v) for k, v in sorted(config["builds"].items()))]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(report_markdown(config, results, header))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"config": config, "results": results}, f, indent=1)
    print("report: %s" % os.path.join(out_dir, "report.md"))
    if args.compare:
        table = compare(load_summary(args.compare), {r["id"]: r for r in results})
        with open(os.path.join(out_dir, "compare.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("# %s vs %s\n\n%s" % (out_dir, args.compare, table))
        print(table)
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
