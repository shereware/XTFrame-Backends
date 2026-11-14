# xt_wgpu_tests (Docs/tasks/T-437.md item 4, Docs/design/10-backends.md §10.10.3): a headless WebGPU
# device for the whole executable. Without an adapter the executable prints `SKIP: no WebGPU adapter`
# and exits 0 (CI runners without a GPU); XT_WGPU_FALLBACK=1 uses a software adapter instead.
# Added by Tests/CMakeLists.txt only when XT_BUILD_WGPU is ON.
add_executable(xt_wgpu_tests
    test_wgpu_main.cpp
    test_wgpu_device.cpp
    test_wgpu_draw.cpp
    test_wgpu_effects.cpp
    test_wgpu_box_shadow.cpp  # T-668
    test_wgpu_mips.cpp  # T-665
    test_wgpu_pack_bc7.cpp  # T-687
    test_wgpu_capabilities.cpp  # T-691
    test_wgpu_affine_plane.cpp  # T-700
    test_wgpu_world_layers.cpp  # T-701
)
target_include_directories(xt_wgpu_tests SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/Code/SDKs/doctest)
target_compile_definitions(xt_wgpu_tests PRIVATE DOCTEST_CONFIG_USE_STD_HEADERS)
# T-670: wgpu_util.h (planInstance, kMaxNonSamplerBindings) is internal to xt_wgpu.
target_include_directories(xt_wgpu_tests PRIVATE ${XT_BACKENDS_SOURCE_DIR}/Code/Runtime/wgpu/src)
# T-438: the draw cases compare with xt::sw (golden_common.h's compareRgba) and compile fixtures.
target_link_libraries(xt_wgpu_tests PRIVATE xt_wgpu xt_gpu xt_core xt_compile xt_sw xt_golden_fixtures)
target_compile_definitions(xt_wgpu_tests PRIVATE XT_GOLDEN_UI_DIR="${PROJECT_SOURCE_DIR}/Tests/golden/ui")
# T-700: test_wgpu_affine_plane.cpp shapes text with the test fonts.
target_compile_definitions(xt_wgpu_tests PRIVATE XT_TEST_FONTS_DIR="${PROJECT_SOURCE_DIR}/Tests/fonts")
# T-687: test_wgpu_pack_bc7.cpp packs a UI Project (xt_bundle) built by Tests/bundle/t687_project.h.
target_link_libraries(xt_wgpu_tests PRIVATE xt_bundle)
target_include_directories(xt_wgpu_tests PRIVATE ${PROJECT_SOURCE_DIR}/Tests/bundle)
# T-691: test_wgpu_capabilities.cpp walks every golden scene fixture, which register themselves by
# static initialization only (see Tests/golden/CMakeLists.txt).
if(XT_MSVC_FRONTEND)
    target_link_options(xt_wgpu_tests PRIVATE /WHOLEARCHIVE:xt_golden_fixtures)
elseif(APPLE)
    target_link_options(xt_wgpu_tests PRIVATE -Wl,-force_load,$<TARGET_FILE:xt_golden_fixtures>)
else()
    target_link_options(xt_wgpu_tests PRIVATE -Wl,--whole-archive $<TARGET_FILE:xt_golden_fixtures>
                                              -Wl,--no-whole-archive)
endif()
xt_enable_exceptions(xt_wgpu_tests)
xt_no_rtti(xt_wgpu_tests)
xt_set_warnings(xt_wgpu_tests)
# wgpu_native.dll next to the executable (Tools/CMake/xt_runtime_output.cmake).
xt_copy_runtime_library(xt_wgpu_tests xt::wgpu_native)

add_test(NAME xt_wgpu_tests COMMAND xt_wgpu_tests)
