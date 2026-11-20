# T-759 (26 §26.3.4, 22 §22.10): xtframe-view draws an `xtframe-pack --atlas` bundle as it draws the
# plain one. Packs SRC twice into OUT (plain, and --atlas --atlas-scope bundle; the atlas pack must
# report ENTRIES cells on one page), takes a headless screenshot of each with RENDERER and requires
# the two files to be byte-identical. Before T-759 the view's own resolver had no atlas cell meta
# and every atlased image drew the whole page.
#
# MAXDIFF / MAXPIXELS (optional, with COMPARE = xt_backends_tests, whose `--compare A B 0` reports the
# differing pixel count and the largest channel difference): instead of byte identity, at most
# MAXPIXELS pixels may differ, by at most MAXDIFF. Only the software renderer's particle scene
# passes them (a known issue, explained in Tests/backends/CMakeLists.txt).
#   cmake -DPACK=<xtframe-pack> -DVIEW=<xtframe-view> -DSRC=<file.xt> -DOUT=<dir> -DRENDERER=sw|gl|wgpu
#         -DENTRIES=<n> [-DCOMPARE=<xt_backends_tests> -DMAXDIFF=<n> -DMAXPIXELS=<n>] -P view_atlas.cmake
file(REMOVE_RECURSE "${OUT}")
file(MAKE_DIRECTORY "${OUT}")
if(ENTRIES STREQUAL "1")
    set(entries_text "1 entry")
else()
    set(entries_text "${ENTRIES} entries")
endif()
foreach(kind plain atlas)
    set(extra "")
    if(kind STREQUAL "atlas")
        set(extra --atlas --atlas-scope bundle)
    endif()
    # T-714: --compress none -- the publish default (auto) would turn the atlas page (>= 256 px) into
    # BC7 while the plain bundle's small images stay PNG; this test compares atlas cells, not BC7.
    execute_process(COMMAND "${PACK}" "${SRC}" -o "${OUT}/${kind}" --no-thumbs --compress none
                            --cache-dir "${OUT}/cache-${kind}" ${extra}
                    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "xtframe-pack ${kind} exit ${rc}: ${out}${err}")
    endif()
    if(kind STREQUAL "atlas" AND NOT "${out}${err}" MATCHES "atlas: 1 page\\(s\\), ${entries_text}")
        message(FATAL_ERROR "the --atlas pack did not put ${entries_text} on one page: ${out}${err}")
    endif()
    execute_process(COMMAND "${VIEW}" "${OUT}/${kind}" --headless --frames 3 --width 320 --height 240
                            --renderer ${RENDERER} --screenshot "${OUT}/${kind}.png"
                    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT rc EQUAL 0 OR NOT EXISTS "${OUT}/${kind}.png")
        message(FATAL_ERROR "xtframe-view ${kind} exit ${rc}: ${out}${err}")
    endif()
endforeach()
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}/plain.png" "${OUT}/atlas.png" RESULT_VARIABLE same)
if(same EQUAL 0)
    message("${RENDERER}: atlas and plain screenshots are byte-identical")
    return()
endif()
if(NOT DEFINED MAXDIFF OR NOT DEFINED MAXPIXELS OR NOT DEFINED COMPARE)
    message(FATAL_ERROR "${RENDERER}: the --atlas bundle's screenshot differs from the plain bundle's "
                        "(${OUT}/atlas.png vs ${OUT}/plain.png)")
endif()
execute_process(COMMAND "${COMPARE}" --compare "${OUT}/plain.png" "${OUT}/atlas.png" 0
                OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${out}" MATCHES ": ([0-9]+) px differ \\([^)]*\\), max channel diff ([0-9]+)")
    message(FATAL_ERROR "${RENDERER}: cannot read the comparison: ${out}${err}")
endif()
set(pixels "${CMAKE_MATCH_1}")
set(worst "${CMAKE_MATCH_2}")
if(pixels GREATER MAXPIXELS OR worst GREATER MAXDIFF)
    message(FATAL_ERROR "${RENDERER}: ${pixels} px differ by up to ${worst} (allowed ${MAXPIXELS} px by ${MAXDIFF}): "
                        "${OUT}/atlas.png vs ${OUT}/plain.png")
endif()
message("${RENDERER}: ${pixels} px differ by up to ${worst} (known issue; allowed ${MAXPIXELS} px by ${MAXDIFF})")
