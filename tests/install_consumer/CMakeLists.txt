# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: MPL-2.0
cmake_minimum_required(VERSION 3.25)
project(alpakaTuneInstallConsumer LANGUAGES CXX)

# There is no FetchContent fallback here: this project only configures against an installed alpakaTune package.
find_package(alpakaTune CONFIG REQUIRED)

add_executable(install_consumer "${CMAKE_CURRENT_LIST_DIR}/../../docs/examples/first_tuner/main.cpp")
target_link_libraries(install_consumer PRIVATE alpakaTune::alpakaTune)
alpaka_finalize(install_consumer)
