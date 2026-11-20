#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

#include "xt/compile/utf8_path.h"
#include "xt/perf.h"
#include "xt/perf_record.h"

/// T-753 (Docs/design/57-measurement.md §57.2 item 3): the second-level zones in xtframe-view. With
/// --perf-detail 2 the run ends with a table of the zones under their top-level phases (per steady frame,
/// i.e. from --warmup on), and --perf-csv <name>.csv also writes <name>.zones.csv: one row per frame, each
/// zone's wall milliseconds and calls in that frame. Header-only; main.cpp is the only includer.

namespace xt::tools::view {

/// The zones of each frame: the difference of PerfTracker::zones() since the previous frame.
class ZoneFrames {
public:
    /// `csvPath` empty: no zone CSV. ok() is false when it could not be opened.
    ZoneFrames(const std::string& csvPath, uint32_t warmup) : m_warmup(warmup) {
        if (csvPath.empty()) {
            return;
        }
        m_csv.open(xt::compile::pathFromUtf8(csvPath), std::ios::binary);
        m_csvWanted = true;
        if (!m_csv) {
            return;
        }
        m_csv << "frame,steady";
        for (uint32_t z = 0; z < xt::kPerfZoneCount; ++z) {
            const char* name = xt::perfZoneCsvName(static_cast<xt::PerfZone>(z));
            m_csv << "," << name << "_ms," << name << "_n";
        }
        m_csv << "\n";
    }
    bool ok() const { return !m_csvWanted || static_cast<bool>(m_csv); }

    /// After markFrameEnd(): the current frame's zones (and its exclusive paint / update / build times).
    void note(const xt::PerfTracker& tracker) {
        if (!tracker.enabled()) {
            return;
        }
        const xt::PerfZoneTotals& now = tracker.zones();
        const xt::FrameStats& f = tracker.current();
        const bool steady = f.frameIndex >= m_warmup;
        if (m_csv) {
            m_csv << f.frameIndex << "," << (steady ? 1 : 0);
        }
        for (uint32_t z = 0; z < xt::kPerfZoneCount; ++z) {
            const double ms = now.ms[z] - m_last.ms[z];
            const uint64_t n = now.count[z] - m_last.count[z];
            if (m_csv) {
                char cell[64];
                std::snprintf(cell, sizeof(cell), ",%.4f,%llu", ms, static_cast<unsigned long long>(n));
                m_csv << cell;
            }
            if (steady) {
                m_sum.ms[z] += ms;
                m_sum.count[z] += n;
            }
        }
        if (m_csv) {
            m_csv << "\n";
        }
        m_last = now;
        if (steady) {
            const xt::FrameRecord r = xt::makeFrameRecord(f);
            m_paint += r.paintMs;
            m_update += r.updateMs;
            m_build += r.buildMs;
            ++m_frames;
        }
    }

    /// The --perf-detail 2 table: each phase, its zones (exclusive: a nested zone's time is not in the
    /// outer one) and the phase time no zone covers.
    void print() const {
        if (m_frames == 0) {
            std::printf("  zones: no steady frames\n");
            return;
        }
        const double inv = 1.0 / static_cast<double>(m_frames);
        std::printf("  zones (--perf-detail 2), per frame over %llu steady frames  %10s %10s\n",
                    static_cast<unsigned long long>(m_frames), "ms", "calls");
        struct Phase {
            const char* name;
            double ms;
        };
        const Phase phases[] = {{"update", m_update}, {"paint", m_paint}, {"build", m_build}};
        for (const Phase& p : phases) {
            std::printf("  %-26s %10.4f\n", p.name, p.ms * inv);
            double covered = 0;
            for (uint32_t z = 0; z < xt::kPerfZoneCount; ++z) {
                const auto zone = static_cast<xt::PerfZone>(z);
                if (std::string(xt::perfZonePhase(zone)) != p.name) {
                    continue;
                }
                covered += m_sum.ms[z];
                std::printf("    %-24s %10.4f %10.1f\n", xt::perfZoneName(zone), m_sum.ms[z] * inv,
                            static_cast<double>(m_sum.count[z]) * inv);
            }
            std::printf("    %-24s %10.4f\n", "(not in a zone)", (p.ms - covered) * inv);
        }
    }

private:
    uint32_t m_warmup = 0;
    bool m_csvWanted = false;
    std::ofstream m_csv;
    xt::PerfZoneTotals m_last;
    xt::PerfZoneTotals m_sum;
    double m_paint = 0;
    double m_update = 0;
    double m_build = 0;
    uint64_t m_frames = 0;
};

/// `<name>.zones.csv` beside --perf-csv `<name>.csv` (or `<file>.zones.csv` without an extension).
inline std::string zonesCsvPath(const std::string& perfCsv) {
    const size_t slash = perfCsv.find_last_of("/\\");
    const size_t dot = perfCsv.find_last_of('.');
    const bool hasExt = dot != std::string::npos && (slash == std::string::npos || dot > slash);
    return (hasExt ? perfCsv.substr(0, dot) : perfCsv) + ".zones.csv";
}

}  // namespace xt::tools::view
