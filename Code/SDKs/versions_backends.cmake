# Third-party versions of the GPU device backends (Docs/design/45-repository-layout.md §45.8 item 4,
# Docs/tasks/T-689.md): the entries of Code/SDKs/versions.cmake that the XTFrame-Runtime mirror does not
# carry; exported with XTFrame-Backends only. Same xt_sdk() syntax as versions.cmake (plain CMake, no
# variables). Included by Tools/CMake/xt_wgpu.cmake; Tools/Scripts/gen_third_party.py reads it after
# versions.cmake when it exists.

# ---- prebuilt binaries (Code/SDKs/_fetched/<name>/<platform>/, Tools/CMake/xt_sdks.cmake xt_sdk_prebuilt)
# Docs/design/10-backends.md §10.10.1 (T-437): wgpu-native's release zips (headers webgpu/webgpu.h +
# webgpu/wgpu.h, the shared library and its import library). A platform without a pair here builds
# without xt_wgpu (XT_BUILD_WGPU turns itself off).
xt_sdk(wgpu-native PREBUILT https://github.com/gfx-rs/wgpu-native v29.0.1.1
    URL_win_x64 https://github.com/gfx-rs/wgpu-native/releases/download/v29.0.1.1/wgpu-windows-x86_64-msvc-release.zip
    SHA256_win_x64 7e67d7445c42aeb85e30f88930fd8d7d83ee769e3390aeb1ada75ebf3cf78132
    URL_linux_x64 https://github.com/gfx-rs/wgpu-native/releases/download/v29.0.1.1/wgpu-linux-x86_64-release.zip
    SHA256_linux_x64 95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a
    URL_mac_arm64 https://github.com/gfx-rs/wgpu-native/releases/download/v29.0.1.1/wgpu-macos-aarch64-release.zip
    SHA256_mac_arm64 a5797a37b1adf720bcd5dcffb291edbbd5b7b14be0a3874c28e6393a655a7a3e
    TITLE "wgpu-native" HOME "https://github.com/gfx-rs/wgpu-native" VERSION "v29.0.1.1 (prebuilt release binaries)"
    LICENSE "MIT / Apache License 2.0 (dual)" USED_BY "`Code/Runtime/wgpu/`"
    PURPOSE "WebGPU implementation (`webgpu.h` + `wgpu.h`, shared library `wgpu_native`) behind the `xt_wgpu` backend")
