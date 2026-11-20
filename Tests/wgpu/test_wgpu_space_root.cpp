// T-760 (Docs/tasks/T-760.md): a perspective space root that is not at the origin draws its own
// background at its own position, once, on the GPU path as on xt::sw (the Painter used to translate
// it a second time by the projection's rootXf).

#include <cstdint>
#include <vector>

#include "doctest.h"
#include "test_wgpu_draw_util.h"
#include "xt/command_list.h"
#include "xt/node.h"
#include "xt/node_arena.h"
#include "xt/painter.h"
#include "xt/property.h"
#include "xt/value.h"

using namespace xt;
using namespace xt::wgpu::test;

namespace {

constexpr int kW = 200, kH = 160;

NodeId makeElement(NodeArena& arena, NodeId parent, float x, float y, float w, float h, Color background) {
    const NodeId id = arena.create(NodeKind::Element);
    if (parent != kInvalidNode) {
        arena.appendChild(parent, id);
    }
    ElementData& data = arena.element(id);
    data.box.width = w;
    data.box.height = h;
    data.style.base[prop::BackgroundColor] = Value::ofColor(background);
    arena.common(id).translate = Vec2{x, y};
    return id;
}

bool near(const uint8_t* p, uint8_t r, uint8_t g, uint8_t b) {
    auto close = [](uint8_t a, uint8_t e) { return (a > e ? a - e : e - a) <= 8; };
    return close(p[0], r) && close(p[1], g) && close(p[2], b);
}

}  // namespace

TEST_CASE("wgpu space root: a perspective root at (60, 60) draws its background there, once (T-760)") {
    NodeArena arena;
    const NodeId root = makeElement(arena, kInvalidNode, 0, 0, kW, kH, Color{1, 1, 1, 1});
    const NodeId stage = makeElement(arena, root, 60, 60, 80, 50, Color{1, 0, 0, 1});
    arena.element(stage).perspectivePx = 300;
    arena.element(stage).perspectiveOriginPx = Vec2{40, 25};
    makeElement(arena, stage, 50, 30, 20, 10, Color{0, 0, 1, 1});
    updateWorldTransforms(arena, root, /*force=*/true);
    CommandList cl = makeList(kW, kH);
    Painter painter;
    painter.paint(arena, root, cl);
    REQUIRE(cl.projections.size() == 1u);

    Session session(kW, kH);
    const std::vector<uint8_t> px = session.render(cl, Color{1, 1, 1, 1});
    CHECK_MESSAGE(session.error.empty(), session.error);
    CHECK(near(pixel(px, kW, 65, 65), 255, 0, 0));        // the stage, at its own position
    CHECK(near(pixel(px, kW, 135, 105), 255, 0, 0));      // its bottom-right corner
    CHECK(near(pixel(px, kW, 120, 95), 0, 0, 255));       // the box inside it
    CHECK(near(pixel(px, kW, 150, 120), 255, 255, 255));  // nothing right of and below it
    CHECK(near(pixel(px, kW, 190, 150), 255, 255, 255));
    CHECK(matches(px, renderSw(cl, kW, kH, Color{1, 1, 1, 1}), kW, kH, "offset space root"));
}
