# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0

# Run clang-tidy with the repository's .clang-tidy while compiling every public header and the configured tests,
# examples and documentation examples. The layout follows alpaka's clangTidy directory.

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "alpakaTune_CLANG_TIDY requires Clang as the C++ compiler.")
endif()

string(REGEX MATCH "^[0-9]+" _alpakaTune_clang_tidy_version "${CMAKE_CXX_COMPILER_VERSION}")
find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy-${_alpakaTune_clang_tidy_version}" clang-tidy REQUIRED)

#---------------------------------------------------------------------------
# Create one source file per public header.
set(_alpakaTune_header_root "${PROJECT_SOURCE_DIR}/include")
file(GLOB_RECURSE _alpakaTune_headers RELATIVE "${_alpakaTune_header_root}" "${_alpakaTune_header_root}/*.hpp")

set(_alpakaTune_generated_root "${CMAKE_CURRENT_BINARY_DIR}/src")
set(_alpakaTune_generated_sources "")
foreach(_header IN LISTS _alpakaTune_headers)
    if(_header MATCHES "^alpakaTune/metrics(\.hpp|/)" AND NOT alpakaTune_DEP_METRICS)
        continue()
    endif()
    set(_source "${_alpakaTune_generated_root}/${_header}.cpp")
    get_filename_component(_source_directory "${_source}" DIRECTORY)
    file(MAKE_DIRECTORY "${_source_directory}")
    file(WRITE "${_source}" "#include <${_header}>\n#include <${_header}>\n")
    list(APPEND _alpakaTune_generated_sources "${_source}")
endforeach()

add_executable(alpakaTune_clang_tidy ${_alpakaTune_generated_sources} src/main.cpp)
target_link_libraries(alpakaTune_clang_tidy PRIVATE alpakaTune::alpakaTune)
if(alpakaTune_DEP_METRICS)
    target_link_libraries(alpakaTune_clang_tidy PRIVATE alpakaTune::metrics)
endif()
alpaka_finalize(alpakaTune_clang_tidy)

#---------------------------------------------------------------------------
# Run clang-tidy while compiling the configured tests, examples and documentation examples.

# Recursively collect the compiled targets below `directory` that build sources from those directories, so
# dependencies (e.g. alpaka, Catch2, yaml-cpp) are not analyzed.
function(alpakaTune_collect_clang_tidy_targets directory out_var)
    set(_result)
    get_property(_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _targets)
        get_target_property(_type ${_target} TYPE)
        if(NOT _type MATCHES "^(EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY)$")
            continue()
        endif()
        get_target_property(_target_sources ${_target} SOURCES)
        get_target_property(_source_dir ${_target} SOURCE_DIR)
        foreach(_source IN LISTS _target_sources)
            # Skip generator expressions, they cannot be resolved at configure time.
            if(_source MATCHES "\\$<")
                continue()
            endif()
            cmake_path(ABSOLUTE_PATH _source BASE_DIRECTORY "${_source_dir}" NORMALIZE)
            file(RELATIVE_PATH _relative_source "${PROJECT_SOURCE_DIR}" "${_source}")
            if(_relative_source MATCHES "^(tests|example|docs/examples)/.*\\.cpp$")
                list(APPEND _result ${_target})
                break()
            endif()
        endforeach()
    endforeach()
    get_property(_subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(_subdirectory IN LISTS _subdirectories)
        alpakatune_collect_clang_tidy_targets("${_subdirectory}" _subdirectory_targets)
        list(APPEND _result ${_subdirectory_targets})
    endforeach()
    set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

alpakatune_collect_clang_tidy_targets("${PROJECT_SOURCE_DIR}" _alpakaTune_clang_tidy_targets)
message(DEBUG "clang-tidy runs during the build of: ${_alpakaTune_clang_tidy_targets}")

set_target_properties(
    alpakaTune_clang_tidy
    ${_alpakaTune_clang_tidy_targets}
    PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE};--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy"
)
set_target_properties(alpakaTune_clang_tidy PROPERTIES FOLDER "clangTidy")
# The test depends on the collected targets, so it cannot pass before they are compiled (and thereby checked).
if(_alpakaTune_clang_tidy_targets)
    add_dependencies(alpakaTune_clang_tidy ${_alpakaTune_clang_tidy_targets})
endif()

add_test(NAME alpakaTune_clang_tidy COMMAND alpakaTune_clang_tidy)
