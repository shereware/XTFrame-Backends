# XTFrame Backends

XTFrame is an MIT-licensed C++20 library for UI documents and vector animation. This repository holds
its **GPU backends** -- the parts that carry a third-party GPU library or a platform graphics API --
on top of the [XTFrame runtime](https://github.com/shereware/XTFrame-Runtime), which it includes as
the git submodule `runtime/`:

- `xt_wgpu`: the WebGPU device over prebuilt [wgpu-native](https://github.com/gfx-rs/wgpu-native)
  (D3D12, Vulkan, Metal);
- `xt_gl`: the OpenGL 3.3 core-profile device;
- `Code/Backends/`: reference platform x renderer glue (`xt_platform_<p>`, `xt_renderer_<r>`,
  `xt_backend_<p>_<r>`), files a host copies into its own project;
- the C ABI's GPU target (`xt_gpu_target_*`, `xt_surface_render_gpu`): the `xtframe` library built here
  is the complete one;
- `xtframe-view` (standalone player: window or headless, screenshots, scripted events) and
  `xtframe-gpu-probe` (GPU frame diagnostics);
- their tests: `Tests/wgpu`, `Tests/gl`, `Tests/backends`, the wgpu golden (`xt_golden --backend wgpu`),
  the C ABI GPU cases and the GPU memory probe (`xt_gpu_mem_probe`, `Tools/Scripts/gpu_mem_matrix.py`).

`xt_core`, the compiler, the software renderer, the device-neutral GPU frame builder `xt_gpu`, bundles,
flow, text and the C ABI header are the runtime's and come only through the submodule. The design
chapters are in `runtime/Docs/design/` (the split is section 45.8 of
`runtime/Docs/design/45-repository-layout.md`; the backends are chapter 10).

> **Read-only snapshot.** This repository is exported from the XTFrame main repository by
> `Tools/Scripts/export_backends.sh`; every commit is one snapshot (`Source-Commit:` trailer). Issues
> are welcome; pull requests are taken as proposals and applied in the main repository.

## Versions

Each tag here has the same name as an XTFrame-Runtime tag, and the `runtime/` submodule of that tag
points to that XTFrame-Runtime tag's commit (intermediate `code.N` tags are paired the same way). Use
one tag for both; never move the submodule to another runtime commit by hand.

## Getting the source

```
git clone --recurse-submodules https://github.com/shereware/XTFrame-Backends.git
# or, in an existing clone:
git submodule update --init
```

## Building

Requires CMake 3.24 or newer and a C++20 compiler. The top-level `CMakeLists.txt` adds `runtime/`
and the runtime's build reads this repository's modules from it, so every option of the runtime
applies (see `runtime/README.md`). The presets here turn on `XT_BUILD_CAPI` (the `xtframe` library).

Windows (Visual Studio 2022):

```
cmake --preset vs2022
cmake --build --preset vs2022-debug
ctest --preset vs2022-debug
```

Linux / macOS (Ninja):

```
cmake --preset ninja-gcc-debug        # or ninja-clang-debug, ninja-*-release
cmake --build --preset ninja-gcc-debug
ctest --preset ninja-gcc-debug
```

The executables, `xtframe` (`xtframe.dll` / `libxtframe.so` / `libxtframe.dylib`) with wgpu-native's
shared library next to it, and the test executables (in `tests/`) land in
`bin/<platform>_<arch>/<Config>/` of this directory, e.g. `bin/win_x64/Release/`. The WebGPU tests
alone: `bash Tools/Scripts/wgpu_tests.sh --preset vs2022-release` (`XT_WGPU_FALLBACK=1` allows a
software adapter, `XT_WGPU_BACKEND=vulkan|d3d12|metal|gl` picks one).

| Option | Default | Effect |
|---|---|---|
| `XT_BUILD_WGPU` | `ON` | `xt_wgpu` (turns itself off with a status line where no wgpu-native archive exists) |
| `XT_BUILD_GL` | `ON` | `xt_gl` |
| `XT_SHELL_GL` | `ON` | `xt_shell` presents through `xt_gl` (`OFF`: `xt_sw` + SDL_Renderer) |
| `XT_BUILD_CAPI` | `ON` in the presets | the `xtframe` library with the GPU target |
| `XT_WGPU_NATIVE_OVERRIDE_DIR` | empty | experimental: a local wgpu-native (`include/webgpu` + `lib`) instead of the pinned one |

## wgpu-native

The pinned version, per-platform download URL and SHA-256 are in `Code/SDKs/versions_backends.cmake`
(currently: win_x64 a prebuilt of upstream commit b83eb64, unmodified source, attached to this
repository's release `wgpu-native-b83eb64`; linux_x64 and mac_arm64 the upstream v29.0.1.1 release
archives). The first
configure downloads the archive of this platform, checks its hash and unpacks it into
`runtime/Code/SDKs/_fetched/wgpu-native/<platform>/` (git-ignored, shared by every build tree);
`-DXT_SDK_OFFLINE=ON` never downloads and expects that directory to be filled already. A platform
without an archive configures without `xt_wgpu` instead of failing.

Memory hints: the b83eb64 build has `WGPUDeviceExtras.memoryHints`, which the configure step detects
(`XT_WGPU_HAS_MEMORY_HINTS`). On win_x64 the GPU target therefore uses the block strategy of
`XT_WGPU_MEMORY_HINTS` (`performance` | `usage` | `manual:<startMiB>:<maxMiB>`; unset = `manual:32:64`)
and logs it when a target is created (`memory hints manual 32..64 MiB (default)`). The v29.0.1.1 builds
of linux_x64 and mac_arm64 have no memory hints and ignore the variable. To try another wgpu-native
commit, build it with `Tools/Scripts/build_wgpu_native_trunk.sh` and point `XT_WGPU_NATIVE_OVERRIDE_DIR`
(or the `vs2022-wgpu-trunk` preset) at it.

## Migrating from XTFrame-Runtime (hosts that use the GPU target)

Until this split, XTFrame-Runtime carried the GPU backends. A host that takes XTFrame-Runtime and uses
`xt_wgpu`, `xt_gl`, `Code/Backends`, the C ABI GPU target or `xtframe-view` moves to this repository:

1. **Source.** Clone XTFrame-Backends with `--recurse-submodules` (or pin it as your submodule /
   lock entry) at the tag you used for XTFrame-Runtime. Its `runtime/` is that same XTFrame-Runtime
   commit, so the runtime sources do not change; drop your separate XTFrame-Runtime checkout.
2. **Build.** Configure this repository's root instead of the runtime's (the presets have the same
   names). A host that adds XTFrame with `add_subdirectory()` adds this directory's
   `runtime/` with `XT_BACKENDS_SOURCE_DIR` set to this directory first:

   ```cmake
   set(XT_BACKENDS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/XTFrame-Backends")
   add_subdirectory("${XT_BACKENDS_SOURCE_DIR}/runtime" xtframe)
   ```

   Without `XT_BACKENDS_SOURCE_DIR` the runtime looks for the backends inside itself, finds none and
   builds without them (one status line each).
3. **Target names** are unchanged: `xt_capi` (output `xtframe`), `xt_wgpu`, `xt_gl`, `xt_gpu`,
   `xt_platform_<p>`, `xt_renderer_<r>`, `xt_backend_<p>_<r>`, `xtframe-view`, `xtframe-gpu-probe`,
   `xt_gpu_mem_probe`. The C ABI header (`runtime/Code/Runtime/capi/include/xt/capi/xtframe.h`), its
   version and its function set are the same in both repositories; only the XTFrame-Runtime build
   answers "built without wgpu" (`xt_gpu_target_create` returns NULL).
4. **Products.** `bin/<platform>_<arch>/<Config>/` of the XTFrame-Backends root (it was the runtime
   root's): `xtframe` with `wgpu_native` next to it, `xtframe-view`, `xtframe-gpu-probe`, and the tests in
   `tests/`. Copy `xtframe` and `wgpu_native` together, as before.
5. **Development options.** `XT_WGPU_NATIVE_OVERRIDE_DIR`, `XT_SDK_OFFLINE`, `XT_BIN_VARIANT` and the
   other cache variables are given to the XTFrame-Backends configure step, e.g.
   `cmake --preset vs2022 -DXT_WGPU_NATIVE_OVERRIDE_DIR=D:/wgpu-native/current`. The fetched sources
   (wgpu-native, SDL2, FreeType, HarfBuzz) are under `runtime/Code/SDKs/_fetched/`; a host that already
   fetched them for XTFrame-Runtime can copy that directory there.

## License

MIT. See [`LICENSE`](LICENSE). Third-party components and their licenses are listed in
[`runtime/Code/SDKs/NOTICE.md`](runtime/Code/SDKs/NOTICE.md); wgpu-native is MIT / Apache License 2.0
(dual).

---

Exported from XTFrame 76eceaf79192 (2026-11-12, version 0.16.0).
