#!/usr/bin/env bash
# Golden order-independence check (Docs/tasks/T-677.md item 4): runs xt_golden once over the whole
# suite, then once per fixture by itself (`--filter =<name>`), and compares each fixture's
# "diffPixels / maxChannelDiff" between the two. Same input must give the same output, so any
# difference means some state of an earlier fixture leaked into a later one (xt_golden keeps one
# renderer for the whole run, as a long-running host does).
# With wgpu it then runs `xt_golden --backend wgpu --order-pairs` over PAIRS (below): every ordered
# pair of those UI fixtures on fresh renderers, the second image byte-identical to its alone image.
#   Tools/Scripts/golden_order_check.sh [--backend sw|wgpu] [--exe <xt_golden>] [--filter <terms>]
#                                       [--pairs <terms> | --no-pairs]
#     --backend  passed to xt_golden (default wgpu: the SW backend keeps no state between fixtures)
#     --exe      the xt_golden executable (default bin/win_x64/Debug/tests/xt_golden.exe)
#     --filter   limit the full-vs-alone runs to these fixtures (xt_golden's --filter syntax)
#     --pairs    the fixtures of the pair check (default PAIRS); --no-pairs skips it
# Prints each fixture whose numbers differ and each differing pair, then
# `ORDER CHECK: PASS (<n> fixtures, <p> pairs)` or `ORDER CHECK: FAIL (...)`; exits 0 / 1, 2 when
# xt_golden ran nothing.
# A fixture that fails its golden compare is still compared (its numbers come from the [FAIL] line).
# One xt_golden process per fixture: a Debug wgpu run takes about two minutes (RTX 4070 Ti, 2026-11-11).
set -u
cd "$(dirname "$0")/../.."

BACKEND=wgpu
EXE=bin/win_x64/Debug/tests/xt_golden.exe
FILTER=""
# The coverage-atlas fixtures (10 §10.11) and the transformed / projected atlas text ones.
PAIRS="=ui_coverage_fill,=ui_coverage_stroke,=ui_coverage_text,=ui_coverage_clip,=ui_coverage_border,=ui_text_transform_skew,=ui_3d_text_perspective,=ui_clip_path"
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --exe) EXE="$2"; shift 2;;
    --filter) FILTER="$2"; shift 2;;
    --pairs) PAIRS="$2"; shift 2;;
    --no-pairs) PAIRS=""; shift;;
    *) echo "golden_order_check: unknown argument '$1'"; exit 2;;
  esac
done
[ -x "$EXE" ] || { echo "golden_order_check: no xt_golden at $EXE"; exit 2; }

# numbers <xt_golden output>: one "name diffPixels maxChannelDiff" line per compared fixture.
numbers() {
  sed -n -E \
    -e 's/^\[PASS\] ([^ ]+) \(diffPixels=([0-9]+) maxChannelDiff=([0-9]+)\).*/\1 \2 \3/p' \
    -e 's/^\[FAIL\] ([^ :]+): ([0-9]+) px differ, max channel diff ([0-9]+).*/\1 \2 \3/p'
}

args=(--backend "$BACKEND")
[ -n "$FILTER" ] && args+=(--filter "$FILTER")
full="$("$EXE" "${args[@]}" 2>&1 </dev/null | numbers)"
[ -n "$full" ] || { echo "golden_order_check: the full run compared no fixture"; exit 2; }

total=0
differ=0
while read -r name fullPx fullMax; do
  [ -n "$name" ] || continue
  total=$((total + 1))
  one="$("$EXE" --backend "$BACKEND" --filter "=$name" 2>&1 </dev/null | numbers | awk -v n="$name" '$1 == n')"
  read -r _ onePx oneMax <<<"$one"
  if [ "${onePx:-x}" != "$fullPx" ] || [ "${oneMax:-x}" != "$fullMax" ]; then
    differ=$((differ + 1))
    echo "DIFFERS $name: full run $fullPx / $fullMax, alone ${onePx:-?} / ${oneMax:-?}"
  fi
done <<<"$full"

pairsMsg=""
pairFail=0
if [ "$BACKEND" = wgpu ] && [ -n "$PAIRS" ]; then
  pairOut="$("$EXE" --backend wgpu --order-pairs --filter "$PAIRS" 2>&1 </dev/null)"
  prc=$?
  grep -E '^\[FAIL\] order ' <<<"$pairOut"
  pairs="$(grep -cE '^\[(PASS|FAIL)\] order ' <<<"$pairOut")"
  bad="$(grep -cE '^\[FAIL\] order ' <<<"$pairOut")"
  if [ "$prc" -ne 0 ] && [ "$bad" = 0 ]; then
    echo "$pairOut" | tail -5
    echo "golden_order_check: the pair check failed to run (exit $prc)"
    exit 2
  fi
  pairsMsg=", $pairs pairs"
  [ "$bad" -gt 0 ] && { pairFail=1; pairsMsg=", $bad of $pairs pairs differ"; }
fi

if [ "$differ" -gt 0 ] || [ "$pairFail" = 1 ]; then
  echo "ORDER CHECK: FAIL ($differ of $total fixtures differ$pairsMsg)"
  exit 1
fi
echo "ORDER CHECK: PASS ($total fixtures$pairsMsg)"
exit 0
