# Third-party versions of the GPU device backends (Docs/design/45-repository-layout.md §45.8 item 4,
# Docs/tasks/T-689.md): the entries of Code/SDKs/versions.cmake that the XTFrame-Runtime mirror does not
# carry; exported with XTFrame-Backends only. Same xt_sdk() syntax as versions.cmake (plain CMake, no
# variables). Included by Tools/CMake/xt_wgpu.cmake; Tools/Scripts/gen_third_party.py reads it after
# versions.cmake when it exists.

# ---- prebuilt binaries (Code/SDKs/_fetched/<name>/<platform>/, Tools/CMake/xt_sdks.cmake xt_sdk_prebuilt)
# Docs/design/10-backends.md §10.10.1 (T-437): wgpu-native's release zips (headers webgpu/webgpu.h +
# webgpu/wgpu.h, the shared library and its import library). A platform without a pair here builds
# without xt_wgpu (XT_BUILD_WGPU turns itself off).
# T-690 (45 §45.8 item 4): win_x64 pins a prebuilt of upstream commit b83eb64 (b83eb6454f908f179a06e149ba212d57a1468ad0,
# unmodified source; has WGPUDeviceExtras.memoryHints, so the default XT_WGPU_MEMORY_HINTS=manual:32:64 applies),
# uploaded to the XTFrame-Backends release `wgpu-native-b83eb64`. linux_x64 and mac_arm64 stay on the upstream
# v29.0.1.1 release archives (no memory hints). The <ref> argument (v29.0.1.1) names the upstream release the
# other platforms use; xt_sdk() has one ref per entry, the per-platform versions are in VERSION.
xt_sdk(wgpu-native PREBUILT https://github.com/gfx-rs/wgpu-native v29.0.1.1
    URL_win_x64 https://github.com/shereware/XTFrame-Backends/releases/download/wgpu-native-b83eb64/wgpu-windows-x86_64-msvc-release-b83eb64.zip
    SHA256_win_x64 cad82e6b03e1d904c0308208739f9cd90c8c0b7c19f709a032b268f2d96fd45e
    URL_linux_x64 https://github.com/gfx-rs/wgpu-native/releases/download/v29.0.1.1/wgpu-linux-x86_64-release.zip
    SHA256_linux_x64 95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a
    URL_mac_arm64 https://github.com/gfx-rs/wgpu-native/releases/download/v29.0.1.1/wgpu-macos-aarch64-release.zip
    SHA256_mac_arm64 a5797a37b1adf720bcd5dcffb291edbbd5b7b14be0a3874c28e6393a655a7a3e
    TITLE "wgpu-native" HOME "https://github.com/gfx-rs/wgpu-native" VERSION "Windows x64: commit b83eb64 (prebuilt, XTFrame-Backends release); Linux x64, macOS arm64: v29.0.1.1 (prebuilt release binaries)"
    LICENSE "MIT / Apache License 2.0 (dual)" USED_BY "`Code/Runtime/wgpu/`"
    PURPOSE "WebGPU implementation (`webgpu.h` + `wgpu.h`, shared library `wgpu_native`) behind the `xt_wgpu` backend")
