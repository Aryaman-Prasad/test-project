cmake_minimum_required(VERSION 3.20)
project(chess_nnue VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer speed, no debug symbols by default
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address,undefined")

# Architecture detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(ARCH_X86 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH_ARM TRUE)
endif()

# SIMD flags
if(ARCH_X86)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -msse4.2 -mpopcnt -mavx2 -mbmi2")
endif()

add_executable(chess_nnue
    src/main.cpp
    src/bitboard.cpp
    src/position.cpp
    src/movegen.cpp
    src/perft.cpp
    # src/search.cpp    — Phase 3
    # src/evaluate.cpp  — Phase 3
    # src/uci.cpp       — Phase 4
)

target_include_directories(chess_nnue PRIVATE src)
