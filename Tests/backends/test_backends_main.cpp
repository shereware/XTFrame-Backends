#define DOCTEST_CONFIG_IMPLEMENT
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "doctest.h"
#include "golden_common.h"
#include "test_temp.h"
#include "test_visual_check.h"
#include "test_watchdog.h"
#include "xt/sw/sw_png.h"

#ifdef _MSC_VER
#include <crtdbg.h>

#include <cstdlib>
#endif

/// xt_backends_tests (Docs/tasks/T-439.md item 8): the Code/Backends parts (platform translation,
/// hidden windows, the three renderers against xt_sw). `--compare A.png B.png` instead compares two
/// screenshots with the GPU/SW tolerance of 15 §15.3 (channel 16, 3 % of the pixels) and exits 0 / 1
/// -- the `xtframe-view --renderer` screenshot tests use it. T-759: `--compare A.png B.png <n>` counts
/// the pixels with a channel difference above `n` instead of 16 (0: every differing pixel); the
/// verdict keeps the 3 % rule, so a caller with its own bound reads the printed numbers.

namespace {

int compareFiles(const char* a, const char* b, int channelTolerance) {
    std::vector<uint8_t> pa, pb;
    int wa = 0, ha = 0, wb = 0, hb = 0;
    if (!xt::sw::readPng(a, pa, wa, ha) || !xt::sw::readPng(b, pb, wb, hb)) {
        std::printf("compare: cannot read '%s' or '%s'\n", a, b);
        return 1;
    }
    if (wa != wb || ha != hb) {
        std::printf("compare: sizes differ (%dx%d vs %dx%d)\n", wa, ha, wb, hb);
        return 1;
    }
    const xt::golden::GoldenResult r =
        xt::golden::compareRgba(pa.data(), pb.data(), wa, ha, channelTolerance, 0.03f);
    const double percent = 100.0 * static_cast<double>(r.diffPixels) / (static_cast<double>(wa) * ha);
    std::printf("compare %s vs %s: %u px differ (%.3f %%), max channel diff %u: %s\n", a, b, r.diffPixels,
                percent, r.maxChannelDiff, r.pass ? "PASS" : "FAIL");
    return r.pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    if ((argc == 4 || argc == 5) && std::strcmp(argv[1], "--compare") == 0) {
        return compareFiles(argv[2], argv[3], argc == 5 ? std::atoi(argv[4]) : 16);
    }
    xt::test::startWatchdog();
    xt::test::enableVisualCheck();  // T-748: the static skip's verify mode in every Context.
    return xt::test::finishTempDir(doctest::Context(argc, argv).run());
}
