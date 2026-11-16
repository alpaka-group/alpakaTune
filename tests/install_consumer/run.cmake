# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0

# Install the already built alpakaTune tree into a scratch prefix, then configure, build and run a consumer that
# finds it with find_package(alpakaTune CONFIG REQUIRED).
#
# cmake -DBUILD_DIR=<alpakaTune build> -DWORK_DIR=<scratch> -DGENERATOR=<generator> -DCXX_COMPILER=<compiler>
#       -DBUILD_TYPE=<type> -P run.cmake

foreach(_required IN ITEMS BUILD_DIR WORK_DIR GENERATOR CXX_COMPILER)
    if(NOT DEFINED ${_required})
        message(FATAL_ERROR "${_required} must be set")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
set(_prefix "${WORK_DIR}/prefix")

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${_prefix}"
    OUTPUT_QUIET
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -S "${CMAKE_CURRENT_LIST_DIR}" -B "${WORK_DIR}/build" -G "${GENERATOR}"
        "-DCMAKE_PREFIX_PATH=${_prefix}" "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}" "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${WORK_DIR}/build" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${WORK_DIR}/build/install_consumer" WORKING_DIRECTORY "${WORK_DIR}" COMMAND_ERROR_IS_FATAL ANY)
