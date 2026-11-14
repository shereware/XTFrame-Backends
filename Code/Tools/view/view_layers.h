#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "view_renderer.h"
#include "xt/context.h"
#include "xt/document.h"
#include "xt/types.h"

/// T-701 (Docs/design/33-world-layers.md, Docs/tasks/T-701.md item 5): `xtframe-view --layers <dir>`
/// and `--layers-view` -- the UI drawn layer by layer (world-layer output), each layer written as a
/// PNG with the list a host needs, and the layers shown standing in space by a small orbit camera of
/// the viewer's own (no engine): the smallest possible host of gpu::WorldLayerBuilder.

namespace xt::view {

struct LayerViewOptions {
    float yawDeg = -30.0f;    ///< the camera's turn around the root box's vertical axis
    float pitchDeg = 14.0f;   ///< and its lift (positive: looking down on the UI)
    float depthScale = 1.0f;  ///< multiplies every layer's z (33 §33.7 item 5); 0 = flat
    float density = 0.0f;     ///< texels per logical px of the layers' textures; 0 = the frame's
};

/// The viewer's camera on the root box: a perspective view orbiting the box's centre, at a distance
/// where the unturned box takes about three quarters of the frame.
class LayerOrbit {
public:
    explicit LayerOrbit(const LayerViewOptions& options) : m_options(options) {}

    /// Root-box point -> homogeneous pixels of a `frameW` x `frameH` frame showing a `boxW` x
    /// `boxH` (logical px) root box.
    Mat4 boxToSurface(float boxW, float boxH, int frameW, int frameH) const;
    /// The ray (root-box coordinates) through the logical window point `pos`.
    void ray(Vec2 pos, float boxW, float boxH, Vec3& origin, Vec3& direction) const;
    /// Takes the input the platform queued out of `context` and puts it back as the layers view
    /// reads it: a right- or middle-button drag turns the camera, the wheel changes the depth
    /// scale, every other pointer event goes in as a ray through the camera (33 §33.6), the rest
    /// as it was.
    void filterInput(Context& context, float boxW, float boxH);

    const LayerViewOptions& options() const { return m_options; }

private:
    struct Frame {
        Vec3 eye, right, down, back;
        float focal = 1.0f;
    };
    Frame frame(float boxW, float boxH) const;

    LayerViewOptions m_options;
    bool m_orbiting = false;
    Vec2 m_lastPos{};
    std::vector<InputEvent> m_events;
};

/// Per-layer bookkeeping of a `--layers` run: how many frames redrew each layer.
struct LayerRedraws {
    uint32_t id = 0;
    uint16_t source = 0;
    uint32_t redraws = 0;
};
/// Adds the `changed` layers of the frame just rendered to `counts`.
void countLayerRedraws(const FrameTarget& target, std::vector<LayerRedraws>& counts);

/// Writes every layer of `target`'s last renderLayers() into `dir` as `layer-<order>-<id>.png`
/// (straight alpha) with `layers.json` (root box, frames, and per layer: id, the element's `id`
/// attribute, order, world3 (row-major, 16), rect, texture size, density, opacity, backfaceHidden,
/// changed, redraws, file), and prints the JSON to stdout. False (with a message on stderr) when a
/// file cannot be written or a layer cannot be read back.
bool dumpLayers(FrameTarget& target, const Document* document, const std::string& dir, Rect rootBox,
                uint32_t frames, const std::vector<LayerRedraws>& redraws);

}  // namespace xt::view
