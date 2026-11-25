# Tests/backends (Docs/tasks/T-439.md item 8, Docs/design/45-repository-layout.md §45.7): Code/Backends.
#   - xt_backends_tests: the platform and renderer parts (event translation, hidden windows, the
#     sw / gl3 / wgpu renderers against xt_sw); `--compare A.png B.png` compares two screenshots.
#   - xt_backend_smoke_<combo>: each xt_backend_* combination linked alone and driven through
#     xt_backend.h (create, pump, frames, destroy); SKIP without a WebGPU adapter.
#   - xtframe-view --renderer: the bogus fallback and the three renderers' headless screenshots,
#     compared pairwise at the GPU/SW tolerance.
# Added by Tests/CMakeLists.txt when xt_platform_sdl2 exists. Tools/Scripts/dev_check.sh suite `backends`.

include("${PROJECT_SOURCE_DIR}/Tests/cmake/expect_exit.cmake")

add_executable(xt_backends_tests
    test_backends_main.cpp
    test_backends_parts.cpp
)
target_include_directories(xt_backends_tests SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/Code/SDKs/doctest)
target_compile_definitions(xt_backends_tests PRIVATE DOCTEST_CONFIG_USE_STD_HEADERS)
target_link_libraries(xt_backends_tests PRIVATE xt_platform_sdl2 xt_renderer_sw xt_sw xt_core xt_golden_fixtures)
if(TARGET xt_platform_win32)
    target_link_libraries(xt_backends_tests PRIVATE xt_platform_win32)
endif()
if(TARGET xt_renderer_gl3)
    target_link_libraries(xt_backends_tests PRIVATE xt_renderer_gl3)
    target_compile_definitions(xt_backends_tests PRIVATE XT_TEST_GL)
endif()
if(TARGET xt_renderer_wgpu)
    target_link_libraries(xt_backends_tests PRIVATE xt_renderer_wgpu)
    target_compile_definitions(xt_backends_tests PRIVATE XT_TEST_WGPU)
    xt_copy_runtime_library(xt_backends_tests xt::wgpu_native)
endif()
xt_enable_exceptions(xt_backends_tests)
xt_no_rtti(xt_backends_tests)
xt_set_warnings(xt_backends_tests)
add_test(NAME xt_backends_tests COMMAND xt_backends_tests)

# One smoke executable per combination this build has; building xt_backends_tests builds them all.
foreach(_combo sdl2_gl3 sdl2_sw sdl2_wgpu win32_wgpu)
    if(TARGET xt_backend_${_combo})
        set(_exe xt_backend_smoke_${_combo})
        add_executable(${_exe} backend_smoke.cpp)
        target_link_libraries(${_exe} PRIVATE xt_backend_${_combo} xt_core)
        target_compile_definitions(${_exe} PRIVATE XT_SMOKE_NAME="${_combo}")
        if(_combo MATCHES "wgpu$")
            xt_copy_runtime_library(${_exe} xt::wgpu_native)
        endif()
        xt_enable_exceptions(${_exe})
        xt_set_warnings(${_exe})
        add_dependencies(xt_backends_tests ${_exe})
        xt_add_expect_exit_test(backend_smoke_${_combo} CODE 0 OUTPUT "smoke ok|SKIP: "
            COMMAND $<TARGET_FILE:${_exe}>)
    endif()
endforeach()
# SDL's offscreen video driver (no display): the software combination still creates and draws.
if(TARGET xt_backend_smoke_sdl2_sw)
    xt_add_expect_exit_test(backend_smoke_sdl2_sw_offscreen CODE 0 OUTPUT "smoke ok"
        COMMAND $<TARGET_FILE:xt_backend_smoke_sdl2_sw>)
    set_tests_properties(backend_smoke_sdl2_sw_offscreen PROPERTIES ENVIRONMENT "SDL_VIDEODRIVER=offscreen")
endif()

if(TARGET xtframe-view)
    set(_view $<TARGET_FILE:xtframe-view>)
    set(_hud "${PROJECT_SOURCE_DIR}/Examples/demos/game-ui/hud/hud.xt")
    # An unknown renderer: one line, the default, exit 0.
    xt_add_expect_exit_test(xtframe-view_renderer_bogus CODE 0
        OUTPUT "unknown renderer 'bogus' \\(built in: sw[a-z ]*\\); using sw.*headless run complete"
        COMMAND ${_view} "${_hud}" --no-flow --headless --frames 1 --renderer bogus)
    xt_add_expect_exit_test(xtframe-view_renderer_version CODE 0 OUTPUT "renderers: sw"
        COMMAND ${_view} --version)
    # T-884: a --renderer window opens the game controllers through xt_shell's GamepadInput (or not,
    # with --no-gamepad); --event axis values still bypass the dead zones. A real controller: untested.
    set(_t884 "${PROJECT_SOURCE_DIR}/Tests/bundle/fixtures/view_input/view_gamepad.xt")
    # Only when the shell presents with gl: --renderer sw is then a FrameTarget, not the shell.
    foreach(_gp gamepad no-gamepad)
        if(NOT XT_SHELL_GL)
            break()
        endif()
        xt_add_expect_exit_test(xtframe-view_t884_renderer_window_${_gp} CODE 0
            OUTPUT "renderer sw \\(sdl2 window\\).*text=0[.]25\n|text=0[.]25\n.*renderer sw \\(sdl2 window\\)"
            COMMAND ${_view} "${_t884}" --renderer sw --hidden --frames 3 --${_gp}
                    --gamepad-deadzone 0.5 0.5 0.5 --event "0 axis pad.rx 0.25" --print-text RxValue
                    --screenshot "${CMAKE_BINARY_DIR}/xtframe-view-t884-${_gp}.png")
    endforeach()

    set(_out "${CMAKE_BINARY_DIR}/xtframe-view-renderer-out")
    file(MAKE_DIRECTORY "${_out}")
    set(_renderers sw)
    if(TARGET xt_renderer_gl3)
        list(APPEND _renderers gl)
    endif()
    if(TARGET xt_renderer_wgpu)
        list(APPEND _renderers wgpu)
    endif()
    # T-666: --msaa 4 -- written when 4 was xtframe-view's default; the GPU shots keep their 4x path.
    foreach(_r IN LISTS _renderers)
        xt_add_expect_exit_test(xtframe-view_renderer_shot_${_r} CODE 0 OUTPUT "wrote screenshot "
            COMMAND ${_view} "${_hud}" --no-flow --headless --frames 2 --width 640 --height 360
                    --renderer ${_r} --msaa 4 --screenshot "${_out}/hud_${_r}.png")
        set_tests_properties(xtframe-view_renderer_shot_${_r} PROPERTIES FIXTURES_SETUP xt_view_renderer_shots)
    endforeach()
    # Pairwise within the GPU/SW tolerance (a renderer that is unavailable at run time falls back
    # to sw with a message, so its screenshot still exists).
    set(_pairs "")
    if(TARGET xt_renderer_wgpu AND TARGET xt_renderer_gl3)
        list(APPEND _pairs wgpu:gl)
    endif()
    if(TARGET xt_renderer_gl3)
        list(APPEND _pairs gl:sw)
    endif()
    if(TARGET xt_renderer_wgpu)
        list(APPEND _pairs wgpu:sw)
    endif()
    foreach(_pair IN LISTS _pairs)
        string(REPLACE ":" ";" _ab "${_pair}")
        list(GET _ab 0 _a)
        list(GET _ab 1 _b)
        xt_add_expect_exit_test(xtframe-view_renderer_compare_${_a}_${_b} CODE 0 OUTPUT ": PASS"
            COMMAND $<TARGET_FILE:xt_backends_tests> --compare "${_out}/hud_${_a}.png" "${_out}/hud_${_b}.png")
        set_tests_properties(xtframe-view_renderer_compare_${_a}_${_b} PROPERTIES
                             FIXTURES_REQUIRED xt_view_renderer_shots)
    endforeach()
    # T-882 (59 §59.4): the stamp's custom effect `executed` comes from each GPU renderer's draw path
    # (gpu::FrameBuilder::setEffectReport): the shader ran (a renderer unavailable at run time falls back
    # to sw, which reports false).
    foreach(_r IN LISTS _renderers)
        if(NOT _r STREQUAL "sw")
            add_test(NAME xtframe-view_capture_gpu_${_r}
                     COMMAND ${CMAKE_COMMAND} -DVIEW=${_view} "-DDOC=${PROJECT_SOURCE_DIR}/Tests/xt/view_capture.xt"
                             -DMODE=gpu -DRENDERER=${_r} "-DOUT=${CMAKE_BINARY_DIR}/xtframe-view-capture-${_r}"
                             -P "${PROJECT_SOURCE_DIR}/Tests/cmake/view_capture.cmake")
            set_tests_properties(xtframe-view_capture_gpu_${_r} PROPERTIES TIMEOUT 300)
        endif()
    endforeach()
    # T-892 (59 §59.4): a shader the wgpu GLSL frontend rejects (an out-of-range float literal, which
    # the compiler accepts) says why in effects[].reason. wgpu only: whether a GL driver rejects that
    # literal is up to the driver (Tests/gl checks the GL message with a broken translation instead).
    if("wgpu" IN_LIST _renderers)
        add_test(NAME xtframe-view_capture_gpu_broken_wgpu
                 COMMAND ${CMAKE_COMMAND} -DVIEW=${_view} "-DDOC=${PROJECT_SOURCE_DIR}/Tests/xt/view_capture_broken.xt"
                         -DMODE=gpu_broken -DRENDERER=wgpu "-DOUT=${CMAKE_BINARY_DIR}/xtframe-view-capture-broken-wgpu"
                         -P "${PROJECT_SOURCE_DIR}/Tests/cmake/view_capture.cmake")
        set_tests_properties(xtframe-view_capture_gpu_broken_wgpu PROPERTIES TIMEOUT 300)
    endif()
    # T-759: an `xtframe-pack --atlas` bundle draws byte-identical to the plain one, per renderer
    # (view_atlas.cmake): atlas_view.xt (<img>, background image, border-image) on every renderer,
    # atlas_view_particles.xt (a textured emitter) on gl and wgpu.
    # Known issue (sw particles only): emitParticles passes the atlas cell's sub-UV as the instance's
    # uv0 / uv1, and the software rasteriser's float rounding of those page coordinates differs from
    # the file's 0..1 by one level on a few pixels (2 px, max 1 when this was written). The sw
    # particle scene therefore allows at most 8 px differing by at most 1; nothing else does.
    if(TARGET xtframe-pack)
        foreach(_r IN LISTS _renderers)
            foreach(_scene atlas_view atlas_view_particles)
                set(_entries 3)
                set(_tolerance "")
                if(_scene STREQUAL "atlas_view_particles")
                    set(_entries 1)
                    if(_r STREQUAL "sw")
                        set(_tolerance -DCOMPARE=$<TARGET_FILE:xt_backends_tests> -DMAXDIFF=1 -DMAXPIXELS=8)
                    endif()
                endif()
                add_test(NAME xtframe-view_${_scene}_${_r}
                         COMMAND ${CMAKE_COMMAND} -DPACK=$<TARGET_FILE:xtframe-pack> -DVIEW=${_view}
                                 -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/atlas_view/${_scene}.xt
                                 -DOUT=${CMAKE_BINARY_DIR}/xtframe-view-${_scene}-${_r} -DRENDERER=${_r}
                                 -DENTRIES=${_entries} ${_tolerance}
                                 -P "${CMAKE_CURRENT_SOURCE_DIR}/view_atlas.cmake")
                set_tests_properties(xtframe-view_${_scene}_${_r} PROPERTIES TIMEOUT 300)
            endforeach()
        endforeach()
    endif()
endif()

# T-713: xtframe-gpu-probe smoke tests over two bundles packed at test time (the cache in the build tree,
# not next to the sources): hud-overlay (a frame that reads the surface back -- `--wgpu` crashed on its
# first frame with "BindGroup with '' label is invalid" before T-713) and the C ABI's flip fixture (a
# flow toggles #card on #card.Click: with --flow the click rebuilds frames, without it nothing changes).
if(TARGET xtframe-gpu-probe AND TARGET xtframe-pack)
    set(_probe_dir "${CMAKE_BINARY_DIR}/gpu-probe-smoke")
    add_test(NAME xtframe-gpu-probe_pack_bundles
             COMMAND ${CMAKE_COMMAND}
                     -DPACK=$<TARGET_FILE:xtframe-pack> -DOUT=${_probe_dir}
                     -DHUD=${PROJECT_SOURCE_DIR}/Examples/demos/hud-overlay
                     -DFLIP=${PROJECT_SOURCE_DIR}/Tests/capi/fixtures/flip
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/gpu_probe_pack.cmake")
    set_tests_properties(xtframe-gpu-probe_pack_bundles PROPERTIES FIXTURES_SETUP xt_gpu_probe_bundles)
    set(_probe $<TARGET_FILE:xtframe-gpu-probe>)
    set(_probe_tests xtframe-gpu-probe_event_no_flow xtframe-gpu-probe_flow_event xtframe-gpu-probe_bad_event)
    xt_add_expect_exit_test(xtframe-gpu-probe_event_no_flow CODE 0 OUTPUT "frames reused 25/25"
        COMMAND ${_probe} "${_probe_dir}/flip" flip 256 256 30 --event "10 #card.Click")
    xt_add_expect_exit_test(xtframe-gpu-probe_flow_event CODE 0
        OUTPUT "flow ui/flip\\.xtflow: loaded.*steady state" REJECT "frames reused 25/25"
        COMMAND ${_probe} "${_probe_dir}/flip" flip 256 256 30 --flow --event "10 #card.Click")
    xt_add_expect_exit_test(xtframe-gpu-probe_bad_event CODE 2 OUTPUT "cannot parse --event 'x #card.Click'"
        COMMAND ${_probe} "${_probe_dir}/flip" flip --event "x #card.Click")
    if(TARGET xt_wgpu)
        list(APPEND _probe_tests xtframe-gpu-probe_wgpu)
        xt_add_expect_exit_test(xtframe-gpu-probe_wgpu CODE 0
            OUTPUT "steady state \\(8 frame\\(s\\), per frame\\):.*render [0-9.]+ ms \\| render \\+ queue wait [0-9.]+ ms.*steady-state tails.*device memory after the last frame: pooled targets [0-9]+"
            COMMAND ${_probe} "${_probe_dir}/hud" hud 640 360 12 --wgpu --warmup 4)
        set_tests_properties(xtframe-gpu-probe_wgpu PROPERTIES SKIP_REGULAR_EXPRESSION "no WebGPU adapter")
    endif()
    set_tests_properties(${_probe_tests} PROPERTIES FIXTURES_REQUIRED xt_gpu_probe_bundles TIMEOUT 300)
endif()
