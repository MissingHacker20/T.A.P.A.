cmake_minimum_required(VERSION 3.20)

project(TAPA VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TAPA_SOURCES
    "Source Code/Features/features.cpp"
    "Source Code/Accumulator/accumulator.cpp"
    "Source Code/Accumulator/pawn_accumulator.cpp"
    "Source Code/Accumulator/piece_accumulator.cpp"
    "Source Code/Features/move.cpp"
    "Source Code/Model/model.cpp"
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

find_package(Threads REQUIRED)
target_link_libraries(tapa_nnue PUBLIC Threads::Threads)

include(CTest)
option(TAPA_BUILD_TESTS "Build and register the TAPA tests" ${BUILD_TESTING})
if(TAPA_BUILD_TESTS)
    add_executable(tapa_tests "Tests/tests.cpp")
    target_link_libraries(tapa_tests PRIVATE tapa_nnue)
    target_include_directories(tapa_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Tests")
    add_test(NAME tapa_features_tests COMMAND tapa_tests)
endif()

option(TAPA_BUILD_BENCHMARKS "Build T.A.P.A. performance benchmarks" OFF)
if(TAPA_BUILD_BENCHMARKS)
    add_executable(tapa_benchmarks "Tests/benchmarks.cpp")
    target_link_libraries(tapa_benchmarks PRIVATE tapa_nnue)
endif()
