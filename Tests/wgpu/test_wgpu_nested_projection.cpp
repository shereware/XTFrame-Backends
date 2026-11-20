// T-761 (Docs/design/19-3d-space.md §19.y.1): nested projections on the wgpu path against xt::sw
// (Tests/common/nested_projection_fixtures.h). Before T-761 the frame builder divided the inner
// levels on the CPU and handed the GPU only the outermost level's w: an image whose perspective sits
// in an inner level was interpolated affinely per triangle.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include "doctest.h"
#include "nested_projection_fixtures.h"
#include "test_wgpu_draw_util.h"
#include "xt/command_list.h"
#include "xt/ir.h"
#include "xt/project3d.h"

using namespace xt;
using namespace xt::wgpu::test;
namespace nested = xt::test::nested;

namespace {

/// Renders `cl` on wgpu and xt::sw, prints the numbers and checks the mean and 15 §15.3 tolerance.
/// Returns the wgpu image.
std::vector<uint8_t> checkNested(const CommandList& cl, const char* what) {
    nested::CheckerResolver resolver;
    IrView view{};
    Session session(nested::kW, nested::kH);
    const std::vector<uint8_t> gpu = session.render(cl, Color{0, 0, 0, 0}, &resolver, &view);
    CHECK_MESSAGE(session.error.empty(), session.error);
    const std::vector<uint8_t> sw = renderSw(cl, nested::kW, nested::kH, Color{0, 0, 0, 0}, &resolver, &view);
    const double mean = nested::meanChannelDiff(gpu, sw);
    const int interior = nested::interiorDiffs(gpu, sw, 16);
    std::printf("wgpu vs sw %s: mean %.3f, %d px differ > 16 off the edges\n", what, mean, interior);
    CHECK(matches(gpu, sw, nested::kW, nested::kH, what));
    CHECK(mean < 1.0);
    CHECK(interior == 0);
    return gpu;
}

/// The screen AABB of `corners` (plane coordinates of the innermost level) projected through the
/// mapping stack of `cl`'s projections and transforms (the CPU path the Painter's bounds and the
/// damage collection use: project3d::projectPoint).
Rect projectedBox(std::span<const project3d::Mapping> stack, std::span<const Vec2> corners) {
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (Vec2 c : corners) {
        bool ok = false;
        const Vec2 s = project3d::projectPoint(stack, Mat2x3{}, c, ok);
        REQUIRE(ok);
        x0 = std::min(x0, s.x);
        y0 = std::min(y0, s.y);
        x1 = std::max(x1, s.x);
        y1 = std::max(y1, s.y);
    }
    return Rect{x0, y0, x1 - x0, y1 - y0};
}

}  // namespace

TEST_CASE("wgpu nested projection: flat outer root and a tilted inner one match SW (T-761)") {
    const CommandList cl = nested::flatOuterTiltedInner();
    const std::vector<uint8_t> gpu = checkNested(cl, "flat outer + tilted inner");

    // The picture stays inside the CPU projection of the image's corners (+1 px of edge coverage):
    // the vertices' x / w, y / w are the points the bounds and damage code computes.
    project3d::Mapping outer;
    outer.hasProjection = true;
    outer.viewProj = cl.projections[0].viewProj;
    outer.rootXf = cl.projections[0].rootXf;
    project3d::Mapping inner;
    inner.hasProjection = true;
    inner.viewProj = cl.projections[1].viewProj;
    inner.rootXf = cl.projections[1].rootXf;
    inner.planeWorld3 = cl.transforms3[1].world3;
    const project3d::Mapping stack[2] = {outer, inner};
    const Rect dst = cl.images[0].dst;
    const Vec2 corners[4] = {Vec2{dst.x, dst.y}, Vec2{dst.x + dst.w, dst.y},
                             Vec2{dst.x + dst.w, dst.y + dst.h}, Vec2{dst.x, dst.y + dst.h}};
    const Rect box = projectedBox(stack, corners);
    int outside = 0;
    for (int y = 0; y < nested::kH; ++y) {
        for (int x = 0; x < nested::kW; ++x) {
            const float fx = static_cast<float>(x);
            const float fy = static_cast<float>(y);
            const bool in = fx + 1.0f >= box.x - 1.0f && fx <= box.x + box.w + 1.0f &&
                            fy + 1.0f >= box.y - 1.0f && fy <= box.y + box.h + 1.0f;
            if (!in && pixel(gpu, nested::kW, x, y)[3] != 0) {
                ++outside;
            }
        }
    }
    CHECK(outside == 0);
}

TEST_CASE("wgpu nested projection: two tilted levels match SW (T-761)") {
    checkNested(nested::bothLevelsTilted(), "both levels tilted");
}

TEST_CASE("wgpu nested projection: an image quad under three levels matches SW (T-761)") {
    checkNested(nested::threeLevels(), "three levels");
}

TEST_CASE("wgpu nested projection: a gradient fill matches SW (T-761)") {
    checkNested(nested::nestedGradientFill(), "nested gradient fill");
}

TEST_CASE("wgpu nested projection: the single-level reference matches SW (T-761)") {
    checkNested(nested::singleLevelTilt(), "single level");
}
