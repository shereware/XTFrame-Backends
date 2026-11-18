# T-713: packs the two bundles of the xtframe-gpu-probe smoke tests (Tests/backends/CMakeLists.txt) into OUT,
# with the pack cache in OUT too, so nothing is written next to the sources.
#   cmake -DPACK=<xtframe-pack> -DOUT=<dir> -DHUD=<hud-overlay project> -DFLIP=<flip fixture> -P gpu_probe_pack.cmake
file(REMOVE_RECURSE "${OUT}")
file(MAKE_DIRECTORY "${OUT}")
foreach(pair "hud|${HUD}" "flip|${FLIP}")
    string(REPLACE "|" ";" parts "${pair}")
    list(GET parts 0 name)
    list(GET parts 1 source)
    execute_process(COMMAND "${PACK}" "${source}" -o "${OUT}/${name}" --no-thumbs --cache-dir "${OUT}/cache-${name}"
                    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "xtframe-pack ${source} exit ${rc}: ${out}${err}")
    endif()
endforeach()
message("packed hud and flip into ${OUT}")
