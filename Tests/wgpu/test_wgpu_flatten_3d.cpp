// T-762 (Docs/design/19-3d-space.md §19.y.2): a grouping flat node inside a 3D space holds its 3D
// children (Tests/common/flatten_3d_fixtures.h) -- the group's opacity, shadow, mask and blend apply
// once to all of it, its children drawn in document order -- on the wgpu path as on xt::sw.

#include <cstdint>
#include <string>
#include <vector>

#include "doctest.h"
#include "flatten_3d_fixtures.h"
#include "test_wgpu_draw_util.h"
#include "xt/command_list.h"

using namespace xt;
using namespace xt::wgpu::test;
namespace fx = xt::test::flatten;

TEST_CASE("wgpu flatten 3D: grouping properties flatten 3D descendants, matching SW (T-762)") {
    std::string error;
    const CommandList cl = fx::paintFixture(error);
    REQUIRE_MESSAGE(error.empty(), error);
    Session session(fx::kW, fx::kH);
    const std::vector<uint8_t> gpu = session.render(cl, Color{1, 1, 1, 1});
    CHECK_MESSAGE(session.error.empty(), session.error);
    const std::vector<uint8_t> sw = renderSw(cl, fx::kW, fx::kH, Color{1, 1, 1, 1});
    CHECK(matches(gpu, sw, fx::kW, fx::kH, "flatten 3D"));
    CHECK(fx::near(fx::at(gpu, 0, fx::kOverlap), 128, 128, 255));
    CHECK(fx::near(fx::at(gpu, 0, fx::kGreenOverRed), 0, 255, 0));
    CHECK(fx::near(fx::at(gpu, 1, fx::kShadowOnly), 0, 0, 0));
    CHECK(fx::near(fx::at(gpu, 2, fx::kBlueOnly), 119, 119, 255));
    CHECK(fx::near(fx::at(gpu, 3, fx::kBlueOnly), 0, 0, 0));
}
