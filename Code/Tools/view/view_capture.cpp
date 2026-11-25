#include "view_capture.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <utility>

#include "xt/compile/utf8_path.h"
#include "xt/effect_utils.h"
#include "xt/sw/sw_png.h"

/// T-882: see view_capture.h.

namespace xt::tools::view {

namespace {

namespace fs = std::filesystem;

/// A number of a flag value: the whole of `text` (no trailing characters), finite.
bool parseNumber(std::string_view text, double& out) {
    const std::string s(text);
    if (s.empty()) {
        return false;
    }
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0' && std::isfinite(out);
}

std::vector<std::string_view> splitCommas(std::string_view text) {
    std::vector<std::string_view> parts;
    size_t start = 0;
    for (;;) {
        const size_t comma = text.find(',', start);
        parts.push_back(text.substr(start, comma == std::string_view::npos ? comma : comma - start));
        if (comma == std::string_view::npos) {
            return parts;
        }
        start = comma + 1;
    }
}

/// The stamp's `kind` of an effect: the `.xt` function name (18 §18.12), "custom", "none".
const char* kindName(EffectKind kind) {
    switch (kind) {
        case EffectKind::DropShadow:
            return "shadow";
        case EffectKind::Glow:
            return "glow";
        case EffectKind::Blur:
            return "blur";
        case EffectKind::ColorAdjust:
            return "color-adjust";
        case EffectKind::GradientMask:
            return "gradient-mask";
        case EffectKind::Shine:
            return "shine";
        case EffectKind::Outline:
            return "outline";
        case EffectKind::BackdropBlur:
            return "backdrop-blur";
        case EffectKind::Custom:
            return "custom";
        case EffectKind::None:
            break;
    }
    return "none";
}

bool overlaps(const Rect& a, const Rect& b) {
    return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

/// The element's border box through its transform (viewport coordinates), outset by its effects' reach.
Rect effectBounds(const Document& doc, const ElementRef& ref) {
    const LayoutBox box = ref.layoutBox();
    const Vec2 corners[4] = {Vec2{0, 0}, Vec2{box.width, 0}, Vec2{box.width, box.height},
                             Vec2{0, box.height}};
    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    for (int i = 0; i < 4; ++i) {
        const Vec2 p = doc.nodeToScreen(ref.node, corners[i]);
        minX = i == 0 ? p.x : std::min(minX, p.x);
        maxX = i == 0 ? p.x : std::max(maxX, p.x);
        minY = i == 0 ? p.y : std::min(minY, p.y);
        maxY = i == 0 ? p.y : std::max(maxY, p.y);
    }
    Rect bounds{minX, minY, maxX - minX, maxY - minY};
    const uint32_t count = ref.effectCount();
    if (count == 0) {
        return bounds;
    }
    // The local -> screen affine map from three projected points (as the editor's effect bounds).
    const Vec2 o = doc.nodeToScreen(ref.node, Vec2{0, 0});
    const Vec2 ax = doc.nodeToScreen(ref.node, Vec2{1, 0});
    const Vec2 ay = doc.nodeToScreen(ref.node, Vec2{0, 1});
    const Mat2x3 xf{ax.x - o.x, ax.y - o.y, ay.x - o.x, ay.y - o.y, o.x, o.y};
    float expand = 0;
    for (uint32_t i = 0; i < count; ++i) {
        expand += effectExpand(effectToScreen(ref.effect(i), xf));
    }
    return Rect{bounds.x - expand, bounds.y - expand, bounds.w + 2 * expand, bounds.h + 2 * expand};
}

/// The element (`#id`, else its tag) and the index among all of its effects that `r` belongs to: the
/// element whose enabled effect `r.stackIndex` has `r`'s kind (and shader) and whose bounds overlap
/// `r.contentPx` most. Empty layer when none matches.
std::pair<std::string, uint32_t> attribute(const Document& doc, const EffectRecord& r) {
    std::string best;
    uint32_t bestIndex = r.stackIndex;
    float bestArea = -1.0f;
    for (uint32_t n = 0;; ++n) {
        const ElementRef ref = doc.elementAt(n);
        if (!ref.valid()) {
            break;
        }
        uint32_t enabled = 0;
        for (uint32_t i = 0; i < ref.effectCount(); ++i) {
            const EffectInstance e = ref.effect(i);
            if (e.enabled == 0) {
                continue;
            }
            if (enabled++ != r.stackIndex) {
                continue;
            }
            if (e.kind != r.kind || (r.kind == EffectKind::Custom && e.shader != r.shader)) {
                break;
            }
            // dpr 1: target pixels are viewport coordinates here (xtframe-view's headless viewport).
            const Rect b = effectBounds(doc, ref);
            const float ix =
                std::min(b.x + b.w, r.contentPx.x + r.contentPx.w) - std::max(b.x, r.contentPx.x);
            const float iy =
                std::min(b.y + b.h, r.contentPx.y + r.contentPx.h) - std::max(b.y, r.contentPx.y);
            const float area = ix > 0 && iy > 0 ? ix * iy : 0.0f;
            if (area > bestArea) {
                bestArea = area;
                best = ref.id().empty() ? std::string(ref.tag()) : "#" + std::string(ref.id());
                bestIndex = i;
            }
            break;
        }
    }
    return {best, bestIndex};
}

std::string countText(uint32_t n, const char* one, const char* many) {
    return std::to_string(n) + (n == 1 ? one : many);
}

}  // namespace

std::string jsonQuote(std::string_view text) {
    std::string out = "\"";
    for (const char c : text) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(c));
                    out += buffer;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
    return out;
}

bool parseCrop(std::string_view text, CropSpec& out) {
    if (!text.empty() && text[0] == '#') {
        if (text.size() == 1) {
            return false;
        }
        out.given = true;
        out.id = std::string(text.substr(1));
        return true;
    }
    const std::vector<std::string_view> parts = splitCommas(text);
    double v[4] = {};
    if (parts.size() != 4) {
        return false;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (!parseNumber(parts[i], v[i])) {
            return false;
        }
    }
    if (!(v[2] > 0.0) || !(v[3] > 0.0)) {
        return false;
    }
    out.given = true;
    out.id.clear();
    out.rect = Rect{static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]),
                    static_cast<float>(v[3])};
    return true;
}

bool parseFrameTimes(std::string_view text, std::vector<double>& out, std::string& error) {
    out.clear();
    for (const std::string_view part : splitCommas(text)) {
        double t = 0;
        if (!parseNumber(part, t) || t < 0.0) {
            error = "--times: expected document clock seconds >= 0, got '" + std::string(part) + "'";
            return false;
        }
        if (!out.empty() && !(t > out.back())) {
            error = "--times: the times must increase ('" + std::string(part) + "' after '" +
                    numberText(out.back()) + "')";
            return false;
        }
        out.push_back(t);
    }
    if (out.size() > kMaxCaptureFrames) {
        error = "--times: at most " + std::to_string(kMaxCaptureFrames) + " time points (got " +
                std::to_string(out.size()) + ")";
        return false;
    }
    return true;
}

std::string frameLabel(double seconds) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.2fs", seconds);
    return buf;
}

std::string numberText(double value) {
    if (!std::isfinite(value)) {
        return "0";
    }
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.6f", value);
    std::string text(buf);
    while (!text.empty() && text.back() == '0') {
        text.pop_back();
    }
    if (!text.empty() && text.back() == '.') {
        text.pop_back();
    }
    return text == "-0" ? "0" : text;
}

bool isJpegPath(std::string_view path) {
    const size_t dot = path.rfind('.');
    if (dot == std::string_view::npos) {
        return false;
    }
    std::string ext(path.substr(dot));
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; });
    return ext == ".jpg" || ext == ".jpeg";
}

std::string framePath(const std::string& screenshot, size_t index) {
    const fs::path shot = compile::pathFromUtf8(screenshot);
    char number[16];
    std::snprintf(number, sizeof number, "_%02u", static_cast<unsigned>(index));
    fs::path name = shot.stem();
    name += number;
    name += isJpegPath(screenshot) ? shot.extension() : fs::path(".png");
    fs::path out = shot;
    out.replace_filename(name);
    return compile::pathToUtf8(out);
}

std::vector<EffectRecord> effectRecords(const std::vector<sw::SwEffectReport>& reports) {
    std::vector<EffectRecord> out;
    for (const sw::SwEffectReport& r : reports) {
        EffectRecord e;
        e.stackIndex = r.stackIndex;
        e.kind = r.kind;
        e.shader = r.shader;
        e.executed = r.executed;
        e.drawnInstead = r.drawnInstead;
        e.clamped = r.clamped;
        e.contentPx = r.contentPx;
        out.push_back(e);
    }
    return out;
}

std::vector<EffectRecord> effectRecords(const std::vector<gpu::FrameBuilder::EffectReport>& reports) {
    std::vector<EffectRecord> out;
    for (const gpu::FrameBuilder::EffectReport& r : reports) {
        EffectRecord e;
        e.stackIndex = r.stackIndex;
        e.kind = EffectKind::Custom;
        e.shader = r.shader;
        e.executed = r.executed;
        e.drawnInstead = r.drawnInstead;
        e.contentPx = r.contentPx;
        e.failure = r.failure;  // T-892
        out.push_back(e);
    }
    return out;
}

void addEffects(Stamp& stamp, const Document* doc, const std::vector<EffectRecord>& records, bool reported,
                const Rect* region) {
    if (!reported) {
        stamp.notes.push_back("effect execution is not reported by this backend");
        return;
    }
    const bool sw = stamp.renderer == "sw";
    if (sw) {
        stamp.notes.push_back("Rendered by the software renderer (xt_sw); custom shaders never run on it.");
    }
    uint32_t ran = 0, fallback = 0, nothing = 0, clamped = 0;
    for (const EffectRecord& r : records) {
        if (region != nullptr && !overlaps(r.contentPx, *region)) {
            continue;  // Outside the captured part.
        }
        Stamp::Effect e;
        e.kind = r.kind;
        e.executed = r.executed;
        e.index = r.stackIndex;
        if (doc != nullptr) {
            const auto [layer, index] = attribute(*doc, r);
            e.layer = layer;
            e.index = index;
        }
        if (r.kind == EffectKind::Custom && r.executed) {
            ++ran;
        } else if (r.kind == EffectKind::Custom) {
            e.hasFallback = true;
            e.fallback = r.drawnInstead;
            if (r.drawnInstead != EffectKind::None) {
                e.reason = sw ? "sw renderer draws the fallback"
                              : stamp.renderer + " renderer could not run the shader; it draws the fallback";
                ++fallback;
            } else {
                e.reason =
                    sw ? "sw renderer draws nothing (fallback: none): not drawn"
                       : stamp.renderer + " renderer could not run the shader (fallback: none): not drawn";
                ++nothing;
            }
            if (!sw) {
                // T-892: the device's compile message replaces the generic sentence when it has one.
                std::string failed = gpu::FrameBuilder::effectFailureReason(stamp.renderer, r.failure);
                if (!failed.empty()) {
                    e.reason = std::move(failed);
                }
            }
        } else if (r.clamped) {
            e.reason = "a radius was clamped to 64 device px (sw renderer)";
            ++clamped;
        }
        stamp.effects.push_back(std::move(e));
    }
    const std::string who = sw ? "SW renderer" : stamp.renderer + " renderer";
    if (ran > 0) {
        stamp.notes.push_back(countText(ran, " custom effect ran", " custom effects ran") + " on the " + who);
    }
    if (fallback > 0) {
        stamp.notes.push_back(countText(fallback, " custom effect was drawn with its fallback (",
                                        " custom effects were drawn with their fallback (") +
                              who + ")");
    }
    if (nothing > 0) {
        stamp.notes.push_back(countText(nothing, " custom effect was", " custom effects were") +
                              " not drawn at all (fallback: none, " + who + ")");
    }
    if (clamped > 0) {
        stamp.notes.push_back(countText(clamped, " effect radius was", " effect radii were") +
                              " clamped to 64 device px (SW renderer)");
    }
}

std::string stampJson(const Stamp& s, bool indented) {
    // One writer for both layouts: `nl(depth)` is a newline and indentation, or nothing.
    auto nl = [indented](int depth) {
        return indented ? "\n" + std::string(static_cast<size_t>(depth) * 2, ' ') : "";
    };
    const std::string colon = indented ? ": " : ":";
    const std::string comma = indented ? ", " : ",";  // between the members of an inner object
    std::string out = "{";
    bool first = true;
    auto key = [&](const char* name) {
        out += first ? "" : ",";
        first = false;
        out += nl(1) + jsonQuote(name) + colon;
    };
    key("renderer");
    out += jsonQuote(s.renderer);
    if (!s.backend.empty()) {
        key("backend");
        out += jsonQuote(s.backend);
    }
    key("source");
    out += jsonQuote("scratch");
    key("device");
    std::string safe;
    if (s.safe.top == s.safe.right && s.safe.top == s.safe.bottom && s.safe.top == s.safe.left) {
        safe = numberText(s.safe.top);
    } else {
        safe = "[" + numberText(s.safe.top) + comma + numberText(s.safe.right) + comma +
               numberText(s.safe.bottom) + comma + numberText(s.safe.left) + "]";
    }
    out += "{" + jsonQuote("w") + colon + numberText(s.w) + comma + jsonQuote("h") + colon + numberText(s.h) +
           comma + jsonQuote("dpr") + colon + numberText(s.dpr) + comma + jsonQuote("uiscale") + colon +
           numberText(s.uiscale) + comma + jsonQuote("safe") + colon + safe + "}";
    key("lang");
    out += jsonQuote(s.lang);
    if (!s.theme.empty()) {
        key("theme");
        out += jsonQuote(s.theme);
    }
    key("clock");
    out += numberText(s.clock);
    key("timelines");
    out += "{";
    for (size_t i = 0; i < s.timelines.size(); ++i) {
        const Stamp::Timeline& t = s.timelines[i];
        out += (i == 0 ? "" : ",") + nl(2) + jsonQuote(t.name) + colon + "{" + jsonQuote("time") + colon +
               numberText(t.time) + comma + jsonQuote("playing") + colon + (t.playing ? "true" : "false") +
               "}";
    }
    out += (s.timelines.empty() ? "" : nl(1)) + "}";
    key("effects");
    out += "[";
    for (size_t i = 0; i < s.effects.size(); ++i) {
        const Stamp::Effect& e = s.effects[i];
        out += (i == 0 ? "" : ",") + nl(2) + "{" + jsonQuote("layer") + colon + jsonQuote(e.layer) + comma +
               jsonQuote("index") + colon + std::to_string(e.index) + comma + jsonQuote("kind") + colon +
               jsonQuote(kindName(e.kind)) + comma + jsonQuote("executed") + colon +
               (e.executed ? "true" : "false");
        if (!e.reason.empty()) {
            out += comma + jsonQuote("reason") + colon + jsonQuote(e.reason);
        }
        if (e.hasFallback) {
            out += comma + jsonQuote("fallback") + colon + jsonQuote(kindName(e.fallback));
        }
        out += "}";
    }
    out += (s.effects.empty() ? "" : nl(1)) + "]";
    key("notes");
    out += "[";
    for (size_t i = 0; i < s.notes.size(); ++i) {
        out += (i == 0 ? "" : ",") + nl(2) + jsonQuote(s.notes[i]);
    }
    out += (s.notes.empty() ? "" : nl(1)) + "]";
    if (!s.frames.empty()) {
        key("frames");
        out += "[";
        for (size_t i = 0; i < s.frames.size(); ++i) {
            const Stamp::Frame& f = s.frames[i];
            out += (i == 0 ? "" : ",") + nl(2) + "{" + jsonQuote("clock") + colon + numberText(f.clock) +
                   comma + jsonQuote("label") + colon + jsonQuote(f.label);
            if (!f.file.empty()) {
                out += comma + jsonQuote("file") + colon + jsonQuote(f.file);
            }
            out += "}";
        }
        out += nl(1) + "]";
    }
    out += nl(0) + "}";
    if (indented) {
        out += "\n";
    }
    return out;
}

bool elementCropRect(const Document& doc, std::string_view id, float dpr, Rect& out) {
    const ElementRef ref = doc.byId(id);
    if (!ref.valid()) {
        return false;
    }
    const Rect b = effectBounds(doc, ref);
    out = Rect{b.x * dpr, b.y * dpr, b.w * dpr, b.h * dpr};
    return true;
}

bool shapeImage(sw::CaptureImage& img, const Rect* crop, int maxEdge, std::string& error) {
    if (crop != nullptr) {
        sw::CaptureImage cut;
        if (sw::crop(img, *crop, cut) != Status::kOk) {
            error = "--crop: the rectangle lies outside the " + std::to_string(img.width) + "x" +
                    std::to_string(img.height) + " picture";
            return false;
        }
        img = std::move(cut);
    }
    if (maxEdge > 0) {
        img = sw::downscaleToMaxEdge(img, maxEdge);
    }
    return true;
}

bool writeImage(const std::string& path, const sw::CaptureImage& img, int jpegQuality,
                std::string& absolute) {
    absolute = compile::pathToUtf8(fs::absolute(compile::pathFromUtf8(path)).lexically_normal());
    std::vector<uint8_t> bytes;
    bool encoded = false;
    if (isJpegPath(path)) {
        sw::CaptureImage flat = img;
        sw::flattenOver(flat, 0, 0, 0);  // As view.capture: transparent pixels over black.
        std::vector<uint8_t> rgb(static_cast<size_t>(flat.width) * static_cast<size_t>(flat.height) * 3);
        for (size_t i = 0, n = rgb.size() / 3; i < n; ++i) {
            rgb[i * 3] = flat.rgba[i * 4];
            rgb[i * 3 + 1] = flat.rgba[i * 4 + 1];
            rgb[i * 3 + 2] = flat.rgba[i * 4 + 2];
        }
        encoded = sw::encodeJpeg(rgb.data(), flat.width, flat.height, jpegQuality, bytes);
    } else {
        encoded = sw::encodePng(img.rgba.data(), img.width, img.height, img.width * 4, bytes);
    }
    std::ofstream file(compile::pathFromUtf8(absolute), std::ios::binary);
    if (!encoded || !file ||
        !file.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size())) ||
        (file.close(), file.fail())) {
        std::fprintf(stderr, "xtframe-view: failed to write screenshot '%s'\n", absolute.c_str());
        return false;
    }
    std::printf("xtframe-view: wrote screenshot %s (%dx%d)\n", absolute.c_str(), img.width, img.height);
    return true;
}

bool writeTextFile(const std::string& path, const std::string& text) {
    std::ofstream file(compile::pathFromUtf8(path), std::ios::binary);
    if (!file || !file.write(text.data(), static_cast<std::streamsize>(text.size()))) {
        std::fprintf(stderr, "xtframe-view: cannot write '%s'\n", path.c_str());
        return false;
    }
    return true;
}

}  // namespace xt::tools::view
