cmake_minimum_required(VERSION 3.20)

project(TAPA VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TAPA_SOURCES
    "Source Code/Features/features.cpp"
    "Source Code/Accumulator/accumulator.cpp"
    "Source Code/Network/network.cpp"
    "Source Code/NNUE/nnue.cpp"
)

add_library(tapa_nnue STATIC ${TAPA_SOURCES})

target_include_directories(tapa_nnue
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/Source Code/Features"
        "${CMAKE_CURRENT_SOURCE_DIR}/Source Code/Accumulator"
        "${CMAKE_CURRENT_SOURCE_DIR}/Source Code/Network"
        "${CMAKE_CURRENT_SOURCE_DIR}/Source Code/NNUE"
)

target_compile_features(tapa_nnue PUBLIC cxx_std_20)

target_compile_options(tapa_nnue PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wpedantic>
)

option(TAPA_BUILD_TESTS "Build the TAPA test sources" OFF)
if(TAPA_BUILD_TESTS)
    add_library(tapa_tests OBJECT "Source Code/Tests/tests.cpp")
    target_link_libraries(tapa_tests PRIVATE tapa_nnue)
    target_include_directories(tapa_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Source Code/Tests")
endif()
