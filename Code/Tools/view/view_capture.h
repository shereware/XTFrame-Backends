#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "xt/document.h"
#include "xt/gpu/frame_builder.h"
#include "xt/layout.h"
#include "xt/sw/capture_util.h"
#include "xt/sw/sw_renderer.h"
#include "xt/types.h"

/// T-882 (Docs/design/59-visual-feedback.md §59.4, §59.8, §59.11 V2): what `xtframe-view --screenshot`
/// adds for visual feedback -- the state stamp (`stamp: <one-line JSON>` on stdout, `--stamp-json FILE`),
/// `--crop`, `--max-edge`, `--times` / `--sheet` and JPEG output. The pixel work is xt::sw's
/// capture_util.h (crop, downscaleToMaxEdge, composeSheet); this file parses the flags, names the
/// files, builds the stamp and writes the images.

namespace xt::tools::view {

/// The most time points one `--times` takes (§59.3 `frames`).
inline constexpr size_t kMaxCaptureFrames = 24;
/// `--jpeg-quality` when not given (§59.3 `format`).
inline constexpr int kDefaultJpegQuality = 80;

/// `--crop`: an element (`#id`, §59.8: its box and its effects' reach) or a pixel rectangle
/// (`x,y,w,h`, output pixels after the dpr).
struct CropSpec {
    bool given = false;
    std::string id;  ///< Without the '#'; empty: `rect`.
    Rect rect{};
};

/// `--crop`'s value into `out` (given = true). False for anything other than `#<id>` or four numbers
/// `x,y,w,h` with w, h > 0.
bool parseCrop(std::string_view text, CropSpec& out);

/// `--times <t1,t2,...>` (T-888; T-882 had it on --frames): document clock seconds, finite, >= 0, strictly
/// increasing, at most kMaxCaptureFrames. False with `error` (the message after "xtframe-view: ") otherwise.
bool parseFrameTimes(std::string_view text, std::vector<double>& out, std::string& error);

/// The contact sheet label of a time point: `<t>s` with two decimals ("0.50s"), as view.capture's.
std::string frameLabel(double seconds);

/// `text` as a JSON string literal, quotes included (T-401's, moved here from main.cpp by T-882).
std::string jsonQuote(std::string_view text);

/// `value` as the shortest decimal of at most six places ("0.983333", "2", "0.5").
std::string numberText(double value);

/// True for a `.jpg` / `.jpeg` path (any case).
bool isJpegPath(std::string_view path);

/// The file of time point `index` (1-based) of `--times` without `--sheet`: `<stem>_<NN><ext>` next to
/// `screenshot` (the extension kept when it is .jpg / .jpeg, else `.png`).
std::string framePath(const std::string& screenshot, size_t index);

/// One effect as a draw path reported it, whichever renderer drew it.
struct EffectRecord {
    uint32_t stackIndex = 0;
    EffectKind kind = EffectKind::None;
    AssetId shader = kInvalidAsset;
    bool executed = false;
    EffectKind drawnInstead = EffectKind::None;  ///< Custom, not executed: the fallback (None: nothing).
    bool clamped = false;                        ///< SW: a radius clamped to 64 device px.
    Rect contentPx{};
    std::string failure;  ///< T-892, GPU: the compile message (EffectReport::failure).
};

std::vector<EffectRecord> effectRecords(const std::vector<sw::SwEffectReport>& reports);
std::vector<EffectRecord> effectRecords(const std::vector<gpu::FrameBuilder::EffectReport>& reports);

/// §59.4's stamp as xtframe-view fills it (no `ui`, `space`, `mode`: those are the editor's).
struct Stamp {
    struct Effect {
        std::string layer;  ///< `#<id>` of the element, else its tag; empty when none matched.
        uint32_t index = 0;
        EffectKind kind = EffectKind::None;
        bool executed = false;
        std::string reason;  ///< Empty: none.
        bool hasFallback = false;
        EffectKind fallback = EffectKind::None;
    };
    struct Timeline {
        std::string name;
        double time = 0;
        bool playing = false;
    };
    struct Frame {
        double clock = 0;
        std::string label;
        std::string file;  ///< Empty: none (a contact sheet).
    };
    std::string renderer;  ///< sw | gl | wgpu: the renderer that drew.
    std::string backend;   ///< Empty: left out (not known).
    float w = 0, h = 0, dpr = 1, uiscale = 1;
    Insets safe{};
    std::string lang;
    std::string theme;  ///< Empty: left out.
    double clock = 0;
    std::vector<Timeline> timelines;
    std::vector<Effect> effects;
    std::vector<std::string> notes;
    std::vector<Frame> frames;  ///< Empty: a single picture (no `frames` member).
};

/// Adds `records` (in `region`, target pixels, when not null) to `stamp.effects`, each attributed to the
/// element of `doc` whose effect `stackIndex` is of the same kind and shader and whose box overlaps the
/// report's content bounds most, and the notes that say what did not run. `reported` false: the
/// renderer does not report its effects -- the note says so and no effect is listed.
void addEffects(Stamp& stamp, const Document* doc, const std::vector<EffectRecord>& records, bool reported,
                const Rect* region);

/// The stamp as JSON: one line (`indented` false, for `stamp: ` on stdout) or indented by two spaces with
/// a final newline (`--stamp-json`).
std::string stampJson(const Stamp& stamp, bool indented);

/// The pixel rectangle `--crop #<id>` cuts: the element's border box through its transform, outset by
/// its effects' reach, times `dpr`. False when `doc` has no element `id`.
bool elementCropRect(const Document& doc, std::string_view id, float dpr, Rect& out);

/// `img` cut to `crop` (when not null) and then shrunk to `maxEdge` (> 0). False with `error` when
/// nothing of `crop` lies inside the picture.
bool shapeImage(sw::CaptureImage& img, const Rect* crop, int maxEdge, std::string& error);

/// Writes `img` to `path` (relative to the current directory) as JPEG at `jpegQuality` when the path
/// ends in .jpg / .jpeg (transparent pixels over black first), else as PNG, and prints `xtframe-view:
/// wrote screenshot <absolute path> (<w>x<h>)`. `absolute` is that path. False after its stderr line.
bool writeImage(const std::string& path, const sw::CaptureImage& img, int jpegQuality, std::string& absolute);

/// Writes `text` to `path` (UTF-8 path). False after its stderr line.
bool writeTextFile(const std::string& path, const std::string& text);

}  // namespace xt::tools::view
