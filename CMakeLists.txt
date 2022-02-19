cmake_minimum_required(VERSION 3.20)
project(transposeV2)

find_package(benchmark REQUIRED)

## PAD Library Interface
add_subdirectory(src)

### mp-media
## debug exe
add_executable(debug.out debug/dataV2Driver.cpp)
target_compile_features(debug.out PRIVATE cxx_std_20)
target_link_libraries(debug.out PRIVATE benchmark::benchmark pad)
target_compile_options(
  debug.out
  PRIVATE
    -Wall
    -march=core-avx2
    -mprefer-vector-width=256
    -static
    -O3
)

### mp-media
## memory performance benchmark
add_executable(memory-benchmark-media benchmarks/memory-benchmark.cpp)
target_compile_features(memory-benchmark-media PRIVATE cxx_std_20)
target_link_libraries(memory-benchmark-media PRIVATE benchmark::benchmark pad)
target_compile_options(
  memory-benchmark-media
  PRIVATE
    -Wall
    -march=core-avx2
    -mprefer-vector-width=256
    -static
    -O3
)