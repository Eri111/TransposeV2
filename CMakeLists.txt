cmake_minimum_required(VERSION 3.20)
project(transposeV2)

find_package(benchmark REQUIRED)

## PAD Library Interface
add_subdirectory(src)

### mp-media
## memory performance benchmark
add_executable(memory-benchmark-media benchmarks/memory-benchmark.cpp)
target_compile_features(memory-benchmark-media PRIVATE cxx_std_20)
target_link_libraries(memory-benchmark-media PRIVATE benchmark::benchmark pad)
target_compile_options(
  memory-benchmark-media
  PRIVATE
    -march=skylake
    -mprefer-vector-width=256
    -static
    -O3
)

## computational performance benchmark
add_executable(computational-benchmark-media benchmarks/computational-benchmark.cpp)
target_compile_features(computational-benchmark-media PRIVATE cxx_std_20)
target_link_libraries(computational-benchmark-media PRIVATE benchmark::benchmark pad)
target_compile_options(
  computational-benchmark-media
  PRIVATE
    -march=core-avx2
    -mprefer-vector-width=256
    -static
    -O3
)

### ziti-rome 
## memory performance benchmark
add_executable(memory-benchmark-rome benchmarks/memory-benchmark.cpp)
target_compile_features(memory-benchmark-rome PRIVATE cxx_std_20)
target_link_libraries(memory-benchmark-rome PRIVATE benchmark::benchmark pad)
target_compile_options(
  memory-benchmark-rome
  PRIVATE
    -march=core-avx2
    -mprefer-vector-width=256
    -static
    -O3
)

## computational performance benchmark
add_executable(computational-benchmark-rome benchmarks/computational-benchmark.cpp)
target_compile_features(computational-benchmark-rome PRIVATE cxx_std_20)
target_link_libraries(computational-benchmark-rome PRIVATE benchmark::benchmark pad)
target_compile_options(
  computational-benchmark-rome
  PRIVATE
    -march=core-avx2
    -mprefer-vector-width=256
    -static
    -O3
)