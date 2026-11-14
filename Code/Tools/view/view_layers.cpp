#include "view_layers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <system_error>

#include "xt/charconv.h"
#include "xt/compile/utf8_path.h"
#include "xt/element_ref.h"
#include "xt/input.h"
#include "xt/sw/sw_png.h"

/// T-701: the layers view of xtframe-view (view_layers.h).

namespace xt::view {

namespace {

constexpr float kDegToRad = 3.14159265358979f / 180.0f;
/// The camera's distance in multiples of the root box's longer side, and how much of the frame
/// the unturned box takes.
constexpr float kDistance = 1.8f;
constexpr float kFill = 0.74f;

Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 normalize(Vec3 v) {
    const float len = std::sqrt(dot(v, v));
    return len > 0.0f ? Vec3{v.x / len, v.y / len, v.z / len} : v;
}

void appendNumber(std::string& out, double value) {
    char buffer[48];
    const size_t n = formatFloat(buffer, sizeof(buffer), value, 9);
    out.append(buffer, n);
}

void appendString(std::string& out, std::string_view text) {
    out += '"';
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(c));
            out += escaped;
        } else {
            out += c;
        }
    }
    out += '"';
}

}  // namespace

LayerOrbit::Frame LayerOrbit::frame(float boxW, float boxH) const {
    // 19 §19.4's camera frame in the y-down, +z-toward-the-viewer space: `back` points from the
    // target to the eye, `right` and `down` span the image.
    const float yaw = m_options.yawDeg * kDegToRad;
    const float pitch = m_options.pitchDeg * kDegToRad;
    Frame f;
    f.back = Vec3{std::sin(yaw) * std::cos(pitch), -std::sin(pitch), std::cos(yaw) * std::cos(pitch)};
    f.right = normalize(cross(Vec3{0, 1, 0}, f.back));
    f.down = cross(f.back, f.right);
    const float distance = kDistance * std::max(boxW, boxH);
    f.eye = Vec3{boxW * 0.5f + f.back.x * distance, boxH * 0.5f + f.back.y * distance, f.back.z * distance};
    f.focal = kFill * distance;
    return f;
}

Mat4 LayerOrbit::boxToSurface(float boxW, float boxH, int frameW, int frameH) const {
    const Frame f = frame(boxW, boxH);
    const float cx = boxW * 0.5f, cy = boxH * 0.5f;
    const float sx = boxW > 0.0f ? static_cast<float>(frameW) / boxW : 1.0f;
    const float sy = boxH > 0.0f ? static_cast<float>(frameH) / boxH : 1.0f;
    // View point v = (right, down, back) . (P - eye); W = -v.z; X = focal v.x + cx W; Y likewise.
    const Vec3 xRow{f.focal * f.right.x - cx * f.back.x, f.focal * f.right.y - cx * f.back.y,
                    f.focal * f.right.z - cx * f.back.z};
    const Vec3 yRow{f.focal * f.down.x - cy * f.back.x, f.focal * f.down.y - cy * f.back.y,
                    f.focal * f.down.z - cy * f.back.z};
    const Vec3 wRow{-f.back.x, -f.back.y, -f.back.z};
    const float depth = m_options.depthScale;
    Mat4 m{};
    m.m[0] = sx * xRow.x;
    m.m[1] = sx * xRow.y;
    m.m[2] = sx * xRow.z * depth;
    m.m[3] = -sx * dot(xRow, f.eye);
    m.m[4] = sy * yRow.x;
    m.m[5] = sy * yRow.y;
    m.m[6] = sy * yRow.z * depth;
    m.m[7] = -sy * dot(yRow, f.eye);
    m.m[10] = 1.0f;
    m.m[12] = wRow.x;
    m.m[13] = wRow.y;
    m.m[14] = wRow.z * depth;
    m.m[15] = -dot(wRow, f.eye);
    return m;
}

void LayerOrbit::ray(Vec2 pos, float boxW, float boxH, Vec3& origin, Vec3& direction) const {
    const Frame f = frame(boxW, boxH);
    const float dx = (pos.x - boxW * 0.5f) / f.focal;
    const float dy = (pos.y - boxH * 0.5f) / f.focal;
    origin = f.eye;
    direction = Vec3{f.right.x * dx + f.down.x * dy - f.back.x, f.right.y * dx + f.down.y * dy - f.back.y,
                     f.right.z * dx + f.down.z * dy - f.back.z};
    // The scene's z is the root box's z times the depth scale.
    const float depth = std::max(m_options.depthScale, 1.0e-3f);
    origin.z /= depth;
    direction.z /= depth;
}

void LayerOrbit::filterInput(Context& context, float boxW, float boxH) {
    m_events.clear();
    context.drainInput(m_events);
    for (const InputEvent& event : m_events) {
        const bool orbitButton =
            event.button == PointerButton::Right || event.button == PointerButton::Middle;
        switch (event.kind) {
            case InputEvent::PointerDown:
                if (orbitButton) {
                    m_orbiting = true;
                    m_lastPos = event.pos;
                    continue;
                }
                break;
            case InputEvent::PointerUp:
                if (orbitButton) {
                    m_orbiting = false;
                    continue;
                }
                break;
            case InputEvent::PointerMove:
                if (m_orbiting) {
                    m_options.yawDeg =
                        std::clamp(m_options.yawDeg - (event.pos.x - m_lastPos.x) * 0.35f, -85.0f, 85.0f);
                    m_options.pitchDeg =
                        std::clamp(m_options.pitchDeg + (event.pos.y - m_lastPos.y) * 0.35f, -70.0f, 70.0f);
                    m_lastPos = event.pos;
                    continue;
                }
                break;
            case InputEvent::Wheel:
                m_options.depthScale =
                    std::clamp(m_options.depthScale * std::pow(1.12f, -event.wheelDelta.y), 0.0f, 12.0f);
                if (m_options.depthScale < 0.02f) {
                    m_options.depthScale = event.wheelDelta.y < 0.0f ? 0.02f : 0.0f;  // down to flat, back up
                }
                continue;
            case InputEvent::PointerLeave:
                m_orbiting = false;
                break;
            default:
                break;
        }
        if (event.kind == InputEvent::PointerMove || event.kind == InputEvent::PointerDown ||
            event.kind == InputEvent::PointerUp) {
            Vec3 origin{}, direction{};
            ray(event.pos, boxW, boxH, origin, direction);
            context.pushPointerRay(event, origin, direction);
        } else {
            context.pushInput(event);
        }
    }
}

void countLayerRedraws(const FrameTarget& target, std::vector<LayerRedraws>& counts) {
    for (const gpu::WorldLayer& layer : target.layers()) {
        LayerRedraws* entry = nullptr;
        for (LayerRedraws& c : counts) {
            if (c.id == layer.id && c.source == layer.source) {
                entry = &c;
                break;
            }
        }
        if (entry == nullptr) {
            counts.push_back(LayerRedraws{layer.id, layer.source, 0});
            entry = &counts.back();
        }
        entry->redraws += layer.changed != 0 ? 1u : 0u;
    }
}

bool dumpLayers(FrameTarget& target, const Document* document, const std::string& dir, Rect rootBox,
                uint32_t frames, const std::vector<LayerRedraws>& redraws) {
    namespace fs = std::filesystem;
    const fs::path root = compile::pathFromUtf8(dir);
    std::error_code ec;
    fs::create_directories(root, ec);
    if (ec) {
        std::fprintf(stderr, "xtframe-view: --layers: cannot create '%s'\n", dir.c_str());
        return false;
    }
    const gpu::WorldLayerBuilder::Stats stats = target.layerStats();
    std::string json = "{\n  \"rootBox\": [";
    appendNumber(json, rootBox.w);
    json += ", ";
    appendNumber(json, rootBox.h);
    json += "],\n  \"frames\": " + std::to_string(frames) +
            ",\n  \"lastFrame\": {\"layers\": " + std::to_string(stats.layers) +
            ", \"built\": " + std::to_string(stats.built) +
            ", \"targetBytes\": " + std::to_string(stats.targetBytes) + "},\n  \"layers\": [";
    std::vector<uint8_t> rgba;
    bool first = true;
    for (const gpu::WorldLayer& layer : target.layers()) {
        if (!target.readLayer(layer, rgba) ||
            rgba.size() != static_cast<size_t>(layer.texWidth) * static_cast<size_t>(layer.texHeight) * 4u) {
            std::fprintf(stderr, "xtframe-view: --layers: cannot read layer %u back\n", layer.id);
            return false;
        }
        const std::string file =
            "layer-" + std::to_string(layer.order) + "-" + std::to_string(layer.id) + ".png";
        const std::string path = compile::pathToUtf8(root / file);
        if (!sw::writePng(path, rgba.data(), layer.texWidth, layer.texHeight, layer.texWidth * 4)) {
            std::fprintf(stderr, "xtframe-view: --layers: cannot write '%s'\n", path.c_str());
            return false;
        }
        std::string_view elementId;
        if (document != nullptr && layer.source == 0 && layer.id != 0 && document->arena().alive(layer.id) &&
            document->arena().kind(layer.id) == NodeKind::Element) {
            elementId =
                ElementRef{const_cast<Document*>(document), layer.id, document->arena().generation(layer.id)}
                    .id();
        }
        uint32_t drawn = 0;
        for (const LayerRedraws& c : redraws) {
            if (c.id == layer.id && c.source == layer.source) {
                drawn = c.redraws;
            }
        }
        json += first ? "\n" : ",\n";
        first = false;
        json += "    {\"id\": " + std::to_string(layer.id) + ", \"element\": ";
        appendString(json, elementId);
        json += ", \"order\": " + std::to_string(layer.order) +
                ", \"source\": " + std::to_string(layer.source) + ",\n     \"world3\": [";
        for (int i = 0; i < 16; ++i) {
            if (i != 0) json += ", ";
            appendNumber(json, layer.world3.m[i]);
        }
        json += "],\n     \"rect\": [";
        const float rect[4] = {layer.rect.x, layer.rect.y, layer.rect.w, layer.rect.h};
        for (int i = 0; i < 4; ++i) {
            if (i != 0) json += ", ";
            appendNumber(json, rect[i]);
        }
        json += "], \"texture\": [" + std::to_string(layer.texWidth) + ", " +
                std::to_string(layer.texHeight) + "], \"density\": ";
        appendNumber(json, layer.density);
        json += ", \"opacity\": ";
        appendNumber(json, layer.opacity);
        json += ",\n     \"backfaceHidden\": " + std::string(layer.backfaceHidden != 0 ? "true" : "false") +
                ", \"stencil\": " + std::string(layer.stencil != 0 ? "true" : "false") +  // T-703
                ", \"changed\": " + std::string(layer.changed != 0 ? "true" : "false") +
                ", \"redraws\": " + std::to_string(drawn) + ", \"file\": ";
        appendString(json, file);
        json += "}";
    }
    json += "\n  ]\n}\n";
    const std::string jsonPath = compile::pathToUtf8(root / "layers.json");
    std::FILE* out = nullptr;
#ifdef _WIN32
    _wfopen_s(&out, (root / "layers.json").c_str(), L"wb");
#else
    out = std::fopen(jsonPath.c_str(), "wb");
#endif
    if (out == nullptr || std::fwrite(json.data(), 1, json.size(), out) != json.size()) {
        if (out != nullptr) std::fclose(out);
        std::fprintf(stderr, "xtframe-view: --layers: cannot write '%s'\n", jsonPath.c_str());
        return false;
    }
    std::fclose(out);
    std::fwrite(json.data(), 1, json.size(), stdout);
    std::printf("xtframe-view: wrote %zu layer(s) to %s\n", target.layers().size(), dir.c_str());
    return true;
}

}  // namespace xt::view
