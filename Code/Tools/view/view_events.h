#pragma once

/// T-713: xtframe-view's `--event` (the parsing and the element / port / pointer / key / gamepad
/// application, T-125 ... T-627; see Code/Tools/view/main.cpp), header-only so xtframe-gpu-probe
/// drives the very same events. `System` events are posted by each tool itself (they go to its
/// flows). Messages start with eventToolName(), "xtframe-view" unless the tool sets its own.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xt/compile/validate.h"
#include "xt/context.h"
#include "xt/document.h"
#include "xt/flow/flow_runtime.h"
#include "xt/input.h"
#include "xt/input_channels.h"

namespace xt::tools::view {

/// The tool name the messages below start with.
inline const char*& eventToolName() {
    static const char* name = "xtframe-view";
    return name;
}

/// One `--event`: a §17.6 element connection point driven just before `frame`'s update().
struct ScriptedEvent {
    uint32_t frame = 0;
    std::string id;      ///< Element id, without the leading '#'; T-175 `Port`: the port name.
    std::string port;    ///< Click / PointerEnter / PointerLeave / Focus / Blur / Change / State / Port.
    std::string value;   ///< `Change` / `Port`: the value to write; `State` (T-174): the state name.
    std::string mode;    ///< `State` only: on / off / toggle.
    uint8_t player = 0;  ///< T-194 `Pad`: the local player.
    uint32_t key = 0;    ///< T-571 `Key`: the xt::KeyCode.
    /// T-627: `Click` and `Pointer`: the mouse button (`button=left|right|middle`, default left).
    xt::PointerButton button = xt::PointerButton::Left;
    float x = 0;                                              ///< T-627 `Pointer`: viewport x.
    float y = 0;                                              ///< T-627 `Pointer`: viewport y.
    xt::InputEvent::Kind kind = xt::InputEvent::PointerMove;  ///< T-627 `Pointer`: move / down / up.
    float wheelX = 0;                                         ///< T-871 `Wheel`: the horizontal notches.
    float wheelY = 0;     ///< T-871 `Wheel`: the vertical notches (positive scrolls down).
    bool hasPos = false;  ///< T-871 `Wheel`: x / y were given (else the root element's centre).
    // T-873 (60 §60.7): `Channel` (id = the canonical channel name), `Axis` (id = pad.<axis>),
    // `KeyDown` / `KeyUp` (key = the KeyCode).
    float channelValue = 0;     ///< `Channel` / `Axis`: the value.
    float channelStrength = 1;  ///< `Channel`: the strength.
    uint8_t channelFlags = 0;   ///< `Channel`: kChannelRelative / kChannelButton.
};

/// T-194: `--event "... press <button>"` button names -> GamepadButton.
inline bool gamepadButtonFromName(const std::string& name, xt::GamepadButton& out) {
    static const std::pair<const char*, xt::GamepadButton> kNames[] = {
        {"A", xt::GamepadButton::A},
        {"B", xt::GamepadButton::B},
        {"X", xt::GamepadButton::X},
        {"Y", xt::GamepadButton::Y},
        {"Back", xt::GamepadButton::Back},
        {"Guide", xt::GamepadButton::Guide},
        {"Start", xt::GamepadButton::Start},
        {"LS", xt::GamepadButton::LeftStick},
        {"RS", xt::GamepadButton::RightStick},
        {"LB", xt::GamepadButton::LeftShoulder},
        {"RB", xt::GamepadButton::RightShoulder},
        {"Up", xt::GamepadButton::DpadUp},
        {"Down", xt::GamepadButton::DpadDown},
        {"Left", xt::GamepadButton::DpadLeft},
        {"Right", xt::GamepadButton::DpadRight},
    };
    for (const auto& n : kNames) {
        if (name == n.first) {
            out = n.second;
            return true;
        }
    }
    return false;
}

/// Splits `text` on ASCII whitespace, dropping empty pieces.
inline std::vector<std::string> splitWhitespace(std::string_view text) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) {
            ++i;
        }
        size_t start = i;
        while (i < text.size() && text[i] != ' ' && text[i] != '\t') {
            ++i;
        }
        if (i > start) {
            out.emplace_back(text.substr(start, i - start));
        }
    }
    return out;
}

/// Parses a decimal frame number; false when `text` is not all digits (or is empty).
inline bool parseFrame(const std::string& text, uint32_t& out) {
    if (text.empty()) {
        return false;
    }
    for (char c : text) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    out = static_cast<uint32_t>(std::strtoul(text.c_str(), nullptr, 10));
    return true;
}

/// T-627: `button=left|right|middle` -> PointerButton; false for any other word.
inline bool parseButtonField(const std::string& word, xt::PointerButton& out) {
    if (word == "button=left") {
        out = xt::PointerButton::Left;
    } else if (word == "button=right") {
        out = xt::PointerButton::Right;
    } else if (word == "button=middle") {
        out = xt::PointerButton::Middle;
    } else {
        return false;
    }
    return true;
}

/// T-627: `move|down|up` -> the pointer InputEvent kind; false for any other word.
inline bool parsePointerKind(const std::string& word, xt::InputEvent::Kind& out) {
    if (word == "move") {
        out = xt::InputEvent::PointerMove;
    } else if (word == "down") {
        out = xt::InputEvent::PointerDown;
    } else if (word == "up") {
        out = xt::InputEvent::PointerUp;
    } else {
        return false;
    }
    return true;
}

/// T-627: a whole-word decimal coordinate (`12`, `-3.5`); false when `text` is not one.
inline bool parseCoordinate(const std::string& text, float& out) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    out = std::strtof(text.c_str(), &end);
    return end == text.c_str() + text.size();
}

/// T-627: a port value with one pair of matching outer quotes ('' or "") removed, so a shell that
/// passes the quotes through (cmd.exe, a spawned process) still sets the empty string.
inline std::string stripOuterQuotes(const std::string& value) {
    if (value.size() >= 2 && (value.front() == '\'' || value.front() == '"') &&
        value.back() == value.front()) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

/// T-392 (XF-078): the one stderr line of a malformed --event: the spec as given, the reason and
/// the element form.
inline void eventSpecError(const std::string& spec, const std::string& reason) {
    std::fprintf(stderr, "%s: cannot parse --event '%s': %s (expected \"<frame> #id.<Event>[=value]\")\n",
                 eventToolName(), spec.c_str(), reason.c_str());
}

/// `--event "<frame> #<id>.<Port>[=<value>]"`; prints its own stderr line and returns false on a
/// malformed spec (the caller then prints the usage and exits 2).
inline bool parseEventSpec(const std::string& spec, ScriptedEvent& out) {
    std::vector<std::string> parts = splitWhitespace(spec);
    {
        // T-194: `[<frame>] press <button> [<player>]`; T-571: `[<frame>] key <name>`.
        size_t at = 0;
        uint32_t frame = 0;
        if (!parts.empty() && parseFrame(parts[0], frame)) {
            at = 1;
        }
        if (parts.size() > at && parts[at] == "press") {
            xt::GamepadButton button{};
            uint32_t player = 0;
            const bool playerOk =
                parts.size() == at + 2 ||
                (parts.size() == at + 3 && parseFrame(parts[at + 2], player) && player < xt::kMaxPlayers);
            if (parts.size() < at + 2 || !gamepadButtonFromName(parts[at + 1], button) || !playerOk) {
                eventSpecError(
                    spec,
                    "a gamepad press is \"[<frame>] press <A|B|X|Y|Back|Guide|Start|LS|RS|LB|RB|Up|"
                    "Down|Left|Right> [<player 0-7>]\"");
                return false;
            }
            out.frame = frame;
            out.port = "Pad";
            out.id = parts[at + 1];
            out.player = static_cast<uint8_t>(player);
            return true;
        }
        // T-627: `[<frame>] pointer <move|down|up> <x> <y> [button=left|right|middle]`.
        if (parts.size() > at && parts[at] == "pointer") {
            xt::InputEvent::Kind kind = xt::InputEvent::PointerMove;
            xt::PointerButton button = xt::PointerButton::Left;
            if (parts.size() < at + 4 || parts.size() > at + 5 || !parsePointerKind(parts[at + 1], kind) ||
                !parseCoordinate(parts[at + 2], out.x) || !parseCoordinate(parts[at + 3], out.y) ||
                (parts.size() == at + 5 &&
                 (kind == xt::InputEvent::PointerMove || !parseButtonField(parts[at + 4], button)))) {
                eventSpecError(spec,
                               "a pointer event is \"[<frame>] pointer <move|down|up> <x> <y> "
                               "[button=left|right|middle]\" (button only with down/up)");
                return false;
            }
            out.frame = frame;
            out.port = "Pointer";
            out.kind = kind;
            out.button = button;
            return true;
        }
        // T-871 (60 §60.7): `[<frame>] wheel <dx> <dy> [<x> <y>]`, notches (positive y scrolls
        // down); without a position the wheel turns over the root element's centre.
        if (parts.size() > at && parts[at] == "wheel") {
            if ((parts.size() != at + 3 && parts.size() != at + 5) ||
                !parseCoordinate(parts[at + 1], out.wheelX) || !parseCoordinate(parts[at + 2], out.wheelY) ||
                (parts.size() == at + 5 &&
                 (!parseCoordinate(parts[at + 3], out.x) || !parseCoordinate(parts[at + 4], out.y)))) {
                eventSpecError(spec, "a wheel event is \"[<frame>] wheel <dx> <dy> [<x> <y>]\" (notches)");
                return false;
            }
            out.frame = frame;
            out.port = "Wheel";
            out.hasPos = parts.size() == at + 5;
            return true;
        }
        // T-873 (60 §60.7): `[<frame>] channel <name> <value> [<strength>] [relative|button]`, one
        // value of an input channel (a `player(<n>).` prefix picks the player).
        if (parts.size() > at && parts[at] == "channel") {
            std::string canonical;
            std::string why;
            size_t end = parts.size();
            if (end > at + 3 && (parts[end - 1] == "relative" || parts[end - 1] == "button")) {
                out.channelFlags = parts[end - 1] == "relative" ? xt::kChannelRelative : xt::kChannelButton;
                --end;
            }
            if (end < at + 3 || end > at + 4 ||
                !xt::parseChannelRef(parts[at + 1], canonical, out.player, &why) ||
                !parseCoordinate(parts[at + 2], out.channelValue) ||
                (end == at + 4 && !parseCoordinate(parts[at + 3], out.channelStrength))) {
                eventSpecError(spec, (why.empty() ? std::string() : why + "; ") +
                                         "a channel value is \"[<frame>] channel <name> <value> [<strength>] "
                                         "[relative|button]\"");
                return false;
            }
            out.frame = frame;
            out.port = "Channel";
            out.id = canonical;
            return true;
        }
        // T-873: `[<frame>] axis <pad.lx|ly|rx|ry|lt|rt> <value>`, one gamepad axis motion (through the
        // existing event: navigation as before, and the pad.* channel).
        if (parts.size() > at && parts[at] == "axis") {
            std::string canonical;
            static const char* const kAxes[] = {"pad.lx", "pad.ly", "pad.rx", "pad.ry", "pad.lt", "pad.rt"};
            int axis = -1;
            if (parts.size() == at + 3 && xt::parseChannelRef(parts[at + 1], canonical, out.player)) {
                for (int i = 0; i < 6; ++i) {
                    axis = canonical == kAxes[i] ? i : axis;
                }
            }
            if (axis < 0 || !parseCoordinate(parts[at + 2], out.channelValue)) {
                eventSpecError(spec,
                               "a gamepad axis is \"[<frame>] axis <pad.lx|ly|rx|ry|lt|rt> <value>\" "
                               "(player(<n>). prefix allowed)");
                return false;
            }
            out.frame = frame;
            out.port = "Axis";
            out.id = canonical;
            out.key = static_cast<uint32_t>(axis);
            return true;
        }
        // T-873: `[<frame>] key down|up <name>`, one key-down or key-up alone.
        if (parts.size() > at + 1 && parts[at] == "key" &&
            (parts[at + 1] == "down" || parts[at + 1] == "up")) {
            uint32_t key = 0;
            if (parts.size() != at + 3 || !xt::compile::keyCodeFromTriggerName(parts[at + 2], key) ||
                key == xt::kTriggerKeyAny) {
                eventSpecError(spec,
                               "a key edge is \"[<frame>] key down|up <name>\" with a key() trigger name");
                return false;
            }
            out.frame = frame;
            out.port = parts[at + 1] == "down" ? "KeyDown" : "KeyUp";
            out.id = parts[at + 2];
            out.key = key;
            return true;
        }
        // T-571 (08 §8.20): `[<frame>] key <name>`, the names of the `key()` trigger.
        if (parts.size() > at && parts[at] == "key") {
            uint32_t key = 0;
            if (parts.size() != at + 2 || !xt::compile::keyCodeFromTriggerName(parts[at + 1], key)) {
                eventSpecError(spec,
                               "a key press is \"[<frame>] key <name>\" with a key() trigger name (Enter, "
                               "Escape, Space, Tab, Backspace, ArrowUp, A-Z, 0-9, F1-F12, ...)");
                return false;
            }
            out.frame = frame;
            out.port = "Key";
            out.id = parts[at + 1];
            out.key = key;
            return true;
        }
    }
    // T-193: `[<frame>] system <kind> [<value>]`.
    const size_t systemAt =
        !parts.empty() && parts[0] == "system" ? 0 : (parts.size() > 1 && parts[1] == "system" ? 1 : 2);
    if (systemAt < 2) {
        if (systemAt == 1 && !parseFrame(parts[0], out.frame)) {
            eventSpecError(spec, "'" + parts[0] + "' is not a frame number");
            return false;
        }
        if (parts.size() < systemAt + 2) {
            eventSpecError(spec, "a system event is \"[<frame>] system <kind> [<value>]\"");
            return false;
        }
        bool known = false;
        for (std::string_view kind : xt::flow::systemEventKinds()) {
            known = known || kind == parts[systemAt + 1];
        }
        if (!known) {
            eventSpecError(spec, "unknown system event '" + parts[systemAt + 1] +
                                     "' (loading, paused, resumed, disconnected, languageChanged, platform, "
                                     "inputDevice)");
            return false;
        }
        out.port = "System";
        out.id = parts[systemAt + 1];
        for (size_t i = systemAt + 2; i < parts.size(); ++i) {
            out.value += (i == systemAt + 2 ? "" : " ") + parts[i];
        }
        return true;
    }
    if (parts.size() >= 4 && parts[1] == "port" && parseFrame(parts[0], out.frame)) {
        // T-175: `<frame> port <name> <value...>` (the value may contain spaces).
        out.port = "Port";
        out.id = parts[2];
        size_t at = spec.find(parts[2], spec.find("port") + 4) + parts[2].size();
        while (at < spec.size() && (spec[at] == ' ' || spec[at] == '\t')) {
            ++at;
        }
        out.value = stripOuterQuotes(spec.substr(at));  // T-627: `port <name> ''` is the empty string.
        return true;
    }
    // T-627: `<frame> #<id>.Click button=left|right|middle`.
    bool hasButton = false;
    if (parts.size() == 3 && parts[2].rfind("button=", 0) == 0) {
        if (!parseButtonField(parts[2], out.button)) {
            eventSpecError(spec, "'" + parts[2] + "' is not button=left|right|middle");
            return false;
        }
        hasButton = true;
        parts.pop_back();
    }
    if (parts.size() != 2) {
        eventSpecError(spec, parts.size() < 2 ? "missing the frame or the target"
                                              : "too many words (no spaces inside the target)");
        return false;
    }
    if (!parseFrame(parts[0], out.frame)) {
        eventSpecError(spec, "'" + parts[0] + "' is not a frame number");
        return false;
    }
    if (parts[1].size() < 2 || parts[1][0] != '#') {
        eventSpecError(spec, "the target '" + parts[1] + "' does not start with '#'");
        return false;
    }
    const std::string& target = parts[1];
    size_t dot = target.find('.', 1);
    if (dot == std::string::npos) {
        eventSpecError(spec, "'" + target + "' has no '.<Event>'");
        return false;
    }
    out.id = target.substr(1, dot - 1);
    std::string port = target.substr(dot + 1);
    size_t eq = port.find('=');
    if (eq != std::string::npos) {
        out.value = port.substr(eq + 1);
        port = port.substr(0, eq);
    }
    if (out.id.empty()) {
        eventSpecError(spec, "'" + target + "' has an empty element id");
        return false;
    }
    if (hasButton && (port != "Click" || eq != std::string::npos)) {
        eventSpecError(spec, "button= applies to Click only");
        return false;
    }
    if (port.rfind("state(", 0) == 0 && port.size() > 7 && port.back() == ')' && eq == std::string::npos) {
        // T-174: `state(<Name>[,on|off|toggle])`.
        std::string inner = port.substr(6, port.size() - 7);
        size_t comma = inner.find(',');
        out.value = inner.substr(0, comma);
        out.mode = comma == std::string::npos ? std::string("on") : inner.substr(comma + 1);
        if (out.value.empty() || (out.mode != "on" && out.mode != "off" && out.mode != "toggle")) {
            eventSpecError(spec, "'" + port + "' is not state(<Name>[,on|off|toggle])");
            return false;
        }
        out.port = "State";
        return true;
    }
    if (port != "Click" && port != "PointerEnter" && port != "PointerLeave" && port != "Focus" &&
        port != "Blur" && port != "Change") {
        eventSpecError(spec, "unknown event '" + port +
                                 "' (Click, PointerEnter, PointerLeave, Focus, Blur, Change, state(<Name>))");
        return false;
    }
    if (port == "Change" && eq == std::string::npos) {
        eventSpecError(spec, "Change needs a '=<value>' suffix");
        return false;
    }
    out.port = std::move(port);
    return true;
}

/// Pushes one raw pointer InputEvent (T-125's --pointer) into `ctx`'s queue; Context::update()
/// dispatches it into the foreground Document before that frame's own update.
inline void pushPointer(xt::Context& ctx, xt::InputEvent::Kind kind, xt::Vec2 pos,
                        xt::PointerButton button = xt::PointerButton::Left) {
    xt::InputEvent event;
    event.kind = kind;
    event.pos = pos;
    event.button = button;  // T-627: --event / --pointer `button=`.
    ctx.pushInput(event);
}

/// T-175: sets declared port `name` from text; one warning when it fails.
inline void setPortText(xt::Document& doc, const std::string& name, const std::string& value) {
    xt::PortId port = doc.lookupPort(name);
    if (port == xt::kInvalidPort) {
        std::fprintf(stderr, "%s: port: no port '%s' in the document\n", eventToolName(), name.c_str());
        return;
    }
    if (doc.setPortText(port, value) != xt::Status::kOk) {
        std::fprintf(stderr, "%s: port: '%s' does not accept '%s'\n", eventToolName(), name.c_str(),
                     value.c_str());
    }
}

/// T-571 (report F84): names the element that took a press meant for `target`: `#<id>` of the
/// nearest node with an id from `hit` up, else `<tag>` of the hit element. Empty when nothing was
/// hit or `hit` is `target` or one of its ancestors (then the target simply has no hit area).
inline std::string coveringElement(xt::Document& doc, const xt::ElementRef& target, xt::NodeId hit) {
    if (hit == xt::kInvalidNode) {
        return std::string();
    }
    for (xt::NodeId up = target.node; up != xt::kInvalidNode; up = doc.arena().parent(up)) {
        if (up == hit) {
            return std::string();
        }
    }
    std::string tag;
    for (xt::NodeId node = hit; node != xt::kInvalidNode; node = doc.arena().parent(node)) {
        if (doc.arena().kind(node) != xt::NodeKind::Element) {
            continue;
        }
        const xt::ElementRef element{&doc, node, doc.arena().generation(node)};
        if (!element.id().empty()) {
            return "#" + std::string(element.id());
        }
        if (tag.empty()) {
            tag = "<" + std::string(element.tag()) + ">";
        }
    }
    return tag;
}

/// Applies one --event against `doc`. Click/PointerEnter/PointerLeave go through Context's input
/// queue (so Document::dispatchInput does the hit testing and the 08.4 state machine);
/// Focus/Blur call Document::setFocus directly and Change calls ElementRef::setValue, whose own
/// Change dispatch is the event. A missing id is reported and skipped.
inline void applyScriptedEvent(xt::Context& ctx, xt::Document& doc, const ScriptedEvent& scripted) {
    if (scripted.port == "Port") {
        setPortText(doc, scripted.id, scripted.value);
        return;
    }
    if (scripted.port == "Pointer") {
        // T-627: one raw pointer event with its button, as --pointer.
        pushPointer(ctx, scripted.kind, xt::Vec2{scripted.x, scripted.y}, scripted.button);
        return;
    }
    if (scripted.port == "Wheel") {
        // T-871 (60 §60.7): one raw wheel event through the input queue, as a mouse wheel would.
        xt::InputEvent ev;
        ev.kind = xt::InputEvent::Wheel;
        ev.wheelDelta = xt::Vec2{scripted.wheelX, scripted.wheelY};
        ev.pos = xt::Vec2{scripted.x, scripted.y};
        if (!scripted.hasPos) {
            const xt::Rect box = doc.root().boundingBox();
            ev.pos = xt::Vec2{box.x + box.w * 0.5f, box.y + box.h * 0.5f};
        }
        ctx.pushInput(ev);
        return;
    }
    if (scripted.port == "Channel") {
        // T-873 (60 §60.7): one channel value through the input queue, as a host would push it.
        xt::InputEvent ev;
        ev.kind = xt::InputEvent::Channel;
        ev.channel = ctx.inputChannel(scripted.id);
        ev.channelValue = scripted.channelValue;
        ev.channelStrength = scripted.channelStrength;
        ev.channelFlags = scripted.channelFlags;
        ev.player = scripted.player;
        ctx.pushInput(ev);
        return;
    }
    if (scripted.port == "Axis") {
        // T-873: one gamepad axis motion of one local player.
        xt::InputEvent ev;
        ev.kind = xt::InputEvent::GamepadAxisMotion;
        ev.gamepadAxis = static_cast<xt::GamepadAxis>(scripted.key);
        ev.axisValue = scripted.channelValue;
        ev.player = scripted.player;
        ctx.pushInput(ev);
        return;
    }
    if (scripted.port == "KeyDown" || scripted.port == "KeyUp") {
        // T-873: a key-down or a key-up alone.
        xt::InputEvent ev;
        ev.kind = scripted.port == "KeyDown" ? xt::InputEvent::KeyDown : xt::InputEvent::KeyUp;
        ev.key = static_cast<xt::KeyCode>(scripted.key);
        ctx.pushInput(ev);
        return;
    }
    if (scripted.port == "Key") {
        // T-571 (08 §8.20): a key press (down + up) through the surface stack, as a keyboard would.
        xt::InputEvent ev;
        ev.kind = xt::InputEvent::KeyDown;
        ev.key = static_cast<xt::KeyCode>(scripted.key);
        ctx.pushInput(ev);
        ev.kind = xt::InputEvent::KeyUp;
        ctx.pushInput(ev);
        return;
    }
    if (scripted.port == "Pad") {
        // T-194: a gamepad button press of one local player, through the surface stack.
        xt::GamepadButton button{};
        gamepadButtonFromName(scripted.id, button);
        xt::InputEvent ev;
        ev.kind = xt::InputEvent::GamepadButtonDown;
        ev.gamepadButton = button;
        ev.player = scripted.player;
        ctx.pushInput(ev);
        ev.kind = xt::InputEvent::GamepadButtonUp;
        ctx.pushInput(ev);
        return;
    }
    xt::ElementRef ref = doc.byId(scripted.id);
    if (!ref.valid()) {
        std::fprintf(stderr, "%s: --event: no element '#%s'\n", eventToolName(), scripted.id.c_str());
        return;
    }
    if (scripted.port == "Focus") {
        doc.setFocus(ref);
        return;
    }
    if (scripted.port == "Blur") {
        doc.setFocus(xt::ElementRef{});
        return;
    }
    if (scripted.port == "Change") {
        ref.setValue(scripted.value);
        return;
    }
    if (scripted.port == "State") {
        const bool on =
            scripted.mode == "toggle" ? !doc.hasCustomState(ref, scripted.value) : scripted.mode == "on";
        if (ctx.setCustomState(ref, scripted.value, on) != xt::Status::kOk) {
            std::fprintf(stderr, "%s: --event: no custom state '%s' in the document\n", eventToolName(),
                         scripted.value.c_str());
        }
        return;
    }
    // The element's border box in viewport coordinates (its layout box through the world
    // transform), whose centre every pointer-driven port aims at.
    xt::Rect box = ref.boundingBox();
    xt::Vec2 center{box.x + box.w * 0.5f, box.y + box.h * 0.5f};
    if (scripted.port == "PointerEnter") {
        pushPointer(ctx, xt::InputEvent::PointerMove, center);
    } else if (scripted.port == "PointerLeave") {
        pushPointer(ctx, xt::InputEvent::PointerMove, xt::Vec2{-1.0f, -1.0f});
    } else {  // Click.
        // T-359: a zero-area element, or one whose centre hits something outside its own subtree,
        // would never receive the pointer click; fire its triggers directly instead.
        bool reachable = box.w > 0.0f && box.h > 0.0f;
        xt::NodeId hitNode = xt::kInvalidNode;
        if (reachable) {
            reachable = false;
            hitNode = doc.hitTestPress(center);
            for (xt::NodeId hit = hitNode; hit != xt::kInvalidNode; hit = doc.arena().parent(hit)) {
                if (hit == ref.node) {
                    reachable = true;
                    break;
                }
            }
        }
        if (!reachable) {
            const std::string cover = coveringElement(doc, ref, hitNode);
            if (!cover.empty()) {
                // T-571 (report F84): another element takes the press at the centre.
                std::fprintf(stderr, "warning: #%s covered by %s at (%g,%g); dispatching click directly\n",
                             scripted.id.c_str(), cover.c_str(), static_cast<double>(center.x),
                             static_cast<double>(center.y));
            } else {
                std::fprintf(stderr, "warning: #%s has no hit area; dispatching click directly\n",
                             scripted.id.c_str());
            }
            (void)doc.fireTrigger(ref, xt::TriggerEvent::Press);
            (void)doc.fireTrigger(ref, xt::TriggerEvent::Release);
            (void)doc.fireTrigger(ref, xt::TriggerEvent::Click);
            return;
        }
        pushPointer(ctx, xt::InputEvent::PointerDown, center, scripted.button);
        pushPointer(ctx, xt::InputEvent::PointerUp, center, scripted.button);
    }
}

}  // namespace xt::tools::view
