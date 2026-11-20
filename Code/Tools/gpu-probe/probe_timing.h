#pragma once

// T-734 (Docs/design/10-backends.md §10.19): `xtframe-gpu-probe --wgpu --timing` -- wall clock and
// main-thread CPU time of every frame phase (wait = wall - CPU), the process's CPU time against the
// main thread's over the steady frames, and the GPU timestamps of xt::wgpu::WgpuRenderer. Included by
// main.cpp only in a build with xt_wgpu; nothing here runs without `--timing`.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <time.h>
#endif

#include "xt/compile/utf8_path.h"
#include "xt/perf_record.h"
#include "xt/thread_clock.h"
#include "xt/wgpu/wgpu_renderer.h"

namespace probe_timing {

/// The phases of one `--wgpu` frame, in loop order.
enum Phase { kUpdate, kPaint, kInflightWait, kRender, kEndFrame, kQueueWait, kPhaseCount };
inline const char* phaseName(int p) {
    static const char* const names[kPhaseCount] = {"update", "paint",     "inflight wait",
                                                   "render", "end frame", "queue wait"};
    return names[p];
}
inline const char* phaseCsvName(int p) {
    static const char* const names[kPhaseCount] = {"update", "paint",     "inflight_wait",
                                                   "render", "end_frame", "queue_wait"};
    return names[p];
}

/// T-750: the thread clock lives in the core now (xt/thread_clock.h); the tracker's own clock is used
/// (PerfTracker::setThreadCpuTiming / threadClock), so the meters below and FrameRecord read one clock.
using ThreadClock = xt::ThreadClock;
inline std::string describeClock(const ThreadClock& clock) {
    char buf[160];
    clock.describe(buf, sizeof(buf));
    return buf;
}

/// The process's CPU time (user + kernel, every thread) and the main thread's (GetThreadTimes, coarse),
/// in milliseconds; false where not measured (non-Windows).
inline bool processTimes(double& processMs, double& threadMs) {
#if defined(_WIN32)
    FILETIME create, exit, kernel, user;
    auto ms = [](const FILETIME& t) {
        return double((uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime) / 10000.0;
    };
    if (!GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user)) return false;
    processMs = ms(kernel) + ms(user);
    if (!GetThreadTimes(GetCurrentThread(), &create, &exit, &kernel, &user)) return false;
    threadMs = ms(kernel) + ms(user);
    return true;
#else
    processMs = threadMs = 0;
    return false;
#endif
}

/// One frame: wall and CPU milliseconds per phase, and its GPU timestamps when they came back.
struct FrameTiming {
    double wall[kPhaseCount] = {};
    double cpu[kPhaseCount] = {};
    bool hasGpu = false;
    xt::wgpu::WgpuGpuTiming gpu;
};

/// Measures one phase: `start` before it, `stop` after.
struct PhaseMeter {
    const ThreadClock* clock = nullptr;
    std::chrono::steady_clock::time_point w0;
    double c0 = 0;
    void start() {
        w0 = std::chrono::steady_clock::now();
        c0 = clock->nowMs();
    }
    void stop(FrameTiming& f, Phase p) const {
        f.cpu[p] = clock->nowMs() - c0;
        f.wall[p] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    }
};

struct Dist {
    double avg = 0, p50 = 0, p99 = 0, max = 0;
};
inline Dist distOf(std::vector<double> v) {
    Dist d;
    if (v.empty()) return d;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    auto at = [&](double q) {
        const size_t k = size_t(std::ceil(q * double(v.size())));
        return v[std::min(v.size() - 1, k > 0 ? k - 1 : 0)];
    };
    d.avg = sum / double(v.size());
    d.p50 = at(0.5);
    d.p99 = at(0.99);
    d.max = v.back();
    return d;
}

/// What the run around the frames knows: steady-window process times, GPU timing state.
struct RunInfo {
    bool finish = true;
    bool gpuOn = false;                  ///< setGpuTiming(true) succeeded
    double periodNs = 0;                 ///< wgpuQueueGetTimestampPeriod
    uint64_t dropped = 0;                ///< WgpuRenderer::gpuTimingDropped
    int gpuMissing = 0;                  ///< steady frames without a GPU result
    bool processOk = false;              ///< processMs / threadMs measured
    double processMs = 0, threadMs = 0;  ///< over the steady window
    std::string clock;                   ///< ThreadClock::describe
};

/// The `timing (T-734, ...)` block over `frames[warmup..]`.
inline void printTiming(const std::vector<FrameTiming>& frames, int warmup, const RunInfo& info) {
    const size_t first = std::min(frames.size(), size_t(std::max(warmup, 0)));
    const size_t n = frames.size() - first;
    std::printf("timing (T-734, %zu steady frames, %s):\n", n, info.finish ? "finish" : "no-finish");
    std::printf("  %-14s %8s %8s %8s %8s | %8s %8s %8s %8s | %8s %8s  (ms)\n", "phase", "wall avg", "p50",
                "p99", "max", "cpu avg", "p50", "p99", "max", "wait avg", "p99");
    int clamped = 0;
    for (int p = 0; p < kPhaseCount; ++p) {
        if (p == kQueueWait && !info.finish) continue;
        std::vector<double> wall, cpu, wait;
        for (size_t i = first; i < frames.size(); ++i) {
            wall.push_back(frames[i].wall[p]);
            cpu.push_back(frames[i].cpu[p]);
            double w = frames[i].wall[p] - frames[i].cpu[p];
            if (w < 0) {
                w = 0;
                ++clamped;
            }
            wait.push_back(w);
        }
        const Dist dw = distOf(wall), dc = distOf(cpu), dt = distOf(wait);
        std::printf("  %-14s %8.3f %8.3f %8.3f %8.3f | %8.3f %8.3f %8.3f %8.3f | %8.3f %8.3f\n", phaseName(p),
                    dw.avg, dw.p50, dw.p99, dw.max, dc.avg, dc.p50, dc.p99, dc.max, dt.avg, dt.p99);
    }
    std::vector<double> span, sum, passes;
    bool encoderLevel = false;
    for (size_t i = first; i < frames.size(); ++i) {
        if (!frames[i].hasGpu) continue;
        span.push_back(frames[i].gpu.spanMs);
        sum.push_back(frames[i].gpu.passSumMs);
        passes.push_back(double(frames[i].gpu.passes));
        encoderLevel = frames[i].gpu.encoderLevel;
    }
    if (info.gpuOn) {
        const Dist ds = distOf(span), dp = distOf(sum), dn = distOf(passes);
        std::printf("  %-14s %8.3f %8.3f %8.3f %8.3f  (%zu frames)\n", "GPU span", ds.avg, ds.p50, ds.p99,
                    ds.max, span.size());
        std::printf("  %-14s %8.3f %8.3f %8.3f %8.3f\n", "GPU pass sum", dp.avg, dp.p50, dp.p99, dp.max);
        std::printf("  %-14s %8.1f %8.0f %8.0f %8.0f\n", "GPU passes", dn.avg, dn.p50, dn.p99, dn.max);
    }
    std::printf("  gpu timing dropped %llu | GPU results missing %d | clamped waits %d\n",
                static_cast<unsigned long long>(info.dropped), info.gpuMissing, clamped);
    const double d = n > 0 ? double(n) : 1.0;
    if (info.processOk) {
        std::printf(
            "  process CPU %.3f ms/frame | main thread CPU %.3f ms/frame (GetThreadTimes, coarse) | other "
            "threads "
            "%.3f ms/frame\n",
            info.processMs / d, info.threadMs / d, (info.processMs - info.threadMs) / d);
    } else {
        std::printf("  process CPU: not measured on this platform\n");
    }
    if (info.gpuOn) {
        std::printf("  %s | GPU: %s, period %.4f ns/tick\n", info.clock.c_str(),
                    span.empty() ? "no results" : (encoderLevel ? "encoder timestamps" : "pass timestamps"),
                    info.periodNs);
    } else {
        std::printf("  %s | GPU: timestamps unsupported (CPU split only)\n", info.clock.c_str());
    }
}

/// T-750: how the meters above agree with FrameRecord (57 §57.2) over the steady frames: each meter
/// phase against the record's phases it spans, and the record's closed budget.
inline void printRecordAgreement(const std::vector<FrameTiming>& frames,
                                 const std::vector<xt::FrameRecord>& records, int warmup, bool finish) {
    const size_t first = std::min(frames.size(), size_t(std::max(warmup, 0)));
    const size_t last = std::min(frames.size(), records.size());
    if (last <= first) return;
    std::vector<double> unaccounted;
    double mUpdate = 0, rUpdate = 0, mPaint = 0, rPaint = 0, mWait = 0, rWait = 0, mRender = 0, rRender = 0,
           mFrame = 0, rFrame = 0;
    for (size_t i = first; i < last; ++i) {
        const FrameTiming& t = frames[i];
        const xt::FrameRecord& r = records[i];
        mUpdate += t.wall[kUpdate];
        rUpdate += double(r.inputMs) + r.updateMs + r.flowMs;
        mPaint += t.wall[kPaint];
        rPaint += double(r.paintMs) + r.damageMs;
        mWait += t.wall[kInflightWait] + (finish ? t.wall[kQueueWait] : 0.0);
        rWait += r.waitMs;
        mRender += t.wall[kRender];
        rRender += double(r.buildMs) + r.submitMs;
        double sum = 0;
        for (int p = 0; p < kPhaseCount; ++p) {
            if (p == kQueueWait && !finish) continue;
            sum += t.wall[p];
        }
        mFrame += sum;
        rFrame += r.frameWallMs;
        unaccounted.push_back(r.unaccountedMs);
    }
    const double d = double(last - first);
    const Dist du = distOf(unaccounted);
    std::printf("frame record (T-750, %zu steady frames; meter vs record, wall ms avg):\n", last - first);
    std::printf("  update  %8.3f vs input + update + flow %8.3f\n", mUpdate / d, rUpdate / d);
    std::printf("  paint   %8.3f vs paint + damage        %8.3f\n", mPaint / d, rPaint / d);
    std::printf("  waits   %8.3f vs wait                  %8.3f\n", mWait / d, rWait / d);
    std::printf("  render  %8.3f vs build + submit        %8.3f\n", mRender / d, rRender / d);
    std::printf("  frame   %8.3f (sum of meters) vs frame_wall %8.3f\n", mFrame / d, rFrame / d);
    std::printf("  unaccounted avg %.3f p99 %.3f ms (%.1f%% of the frame)\n", du.avg, du.p99,
                rFrame > 0 ? 100.0 * du.avg / (rFrame / d) : 0.0);
}

/// `--timing-csv` (T-750): one row per frame -- FrameRecord v1 through the runtime's column table
/// (`steady` = 1 after the warm-up), then the probe's own meters and GPU timestamps as `host.*` columns.
inline bool writeCsv(const std::string& path, const std::vector<FrameTiming>& frames,
                     const std::vector<xt::FrameRecord>& records, int warmup) {
    std::ofstream out(xt::compile::pathFromUtf8(path), std::ios::binary);
    if (!out) return false;
    std::vector<char> line(16384);
    auto name = [](int p, const char* unit) { return std::string("host.") + phaseCsvName(p) + unit; };
    size_t n = xt::frameRecordCsvHeader(line.data(), line.size());
    for (int p = 0; p < kPhaseCount; ++p) {
        n = xt::csvAppendText(line.data(), line.size(), n, name(p, "_wall_ms").c_str());
        n = xt::csvAppendText(line.data(), line.size(), n, name(p, "_cpu_ms").c_str());
    }
    n = xt::csvAppendText(line.data(), line.size(), n, "host.gpu_span_ms");
    n = xt::csvAppendText(line.data(), line.size(), n, "host.gpu_pass_sum_ms");
    n = xt::csvAppendText(line.data(), line.size(), n, "host.gpu_passes");
    out.write(line.data(), std::streamsize(std::min(n, line.size() - 1)));
    out << '\n';
    const size_t count = std::min(frames.size(), records.size());
    for (size_t i = 0; i < count; ++i) {
        const FrameTiming& t = frames[i];
        xt::FrameRecord r = records[i];
        r.steady = int(i) >= warmup ? 1u : 0u;
        n = xt::frameRecordCsvRow(r, line.data(), line.size());
        for (int p = 0; p < kPhaseCount; ++p) {
            n = xt::csvAppendNumber(line.data(), line.size(), n, t.wall[p], 4);
            n = xt::csvAppendNumber(line.data(), line.size(), n, t.cpu[p], 4);
        }
        if (t.hasGpu) {
            n = xt::csvAppendNumber(line.data(), line.size(), n, t.gpu.spanMs, 4);
            n = xt::csvAppendNumber(line.data(), line.size(), n, t.gpu.passSumMs, 4);
            n = xt::csvAppendNumber(line.data(), line.size(), n, double(t.gpu.passes), 0);
        } else {
            for (int k = 0; k < 3; ++k) n = xt::csvAppendText(line.data(), line.size(), n, "");
        }
        out.write(line.data(), std::streamsize(std::min(n, line.size() - 1)));
        out << '\n';
    }
    out.close();
    return !out.fail();
}

}  // namespace probe_timing
