#include <benchmark/benchmark.h>
#include <iostream>
#include <oneapi/tbb.h>

#define DATA_POLICY static
#define TRANSPOSE_POLICY static

using DataPartitioner = oneapi::tbb::simple_partitioner;
static DataPartitioner dataPart;

using TransposePartitioner = oneapi::tbb::simple_partitioner;
static TransposePartitioner transPart;

using InValType = float;
using OutValType = bool;

#include "dataV2.hpp"
#include "transpose.hpp"
#include "pinningobserver.hpp"


// constexpr size_t simd_width = 32;
constexpr bool useSerInit = false;
constexpr bool do_verify = false;


// bool floatEquals(double lhs, double rhs, double epsilon = 1e-5) {
//     return std::abs(lhs - rhs) < epsilon;
// }

static void BenchmarkArguments(benchmark::internal::Benchmark* b) {
	const ssize_t lowerLimit = 8;
	const ssize_t upperLimit = 8;

	const ssize_t lowerGS = 8;
	const ssize_t upperGS = 8;
	for (auto j = lowerGS; j <= upperGS; j += 8)
	{	
		for (auto i = lowerLimit; i <= upperLimit; ++i)
		{
			// b->Args({(2*i), (i), (j)}); // first arg in_rows (Spalten ANZAHL), second arg in_columns
			b->Args({(1 << i), (1 << i), (j)}); // first arg in_rows (Spalten ANZAHL), second arg in_columns
			// b->Args({(i), (i), (j)}); // first arg in_rows (Spalten ANZAHL), second arg in_columns
		}
	}
}

void transposeCustomCounter(benchmark::State& state) {
	state.counters["Elements"] = state.range(0) * state.range(1);
	state.counters["Bytes_processed"] = 2 * state.range(1) * state.range(0) * sizeof(InValType);
	state.counters["Matrix_height"] = (int)state.range(0);
	state.counters["Matrix_width"] = (int)state.range(1);
	state.counters["Tile_width"] = (int)state.range(2);
}

static void Serial(benchmark::State& state){
	std::vector<OutValType, pad::default_init_allocator<OutValType>> out_mat(state.range(0)*state.range(1));
    std::fill(out_mat.begin(), out_mat.end(), (OutValType)0.0);

	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	auto beginB = out_mat.begin();
	auto endB = out_mat.end();

	for (auto _ : state) {
		transpose::transposeSerial(beginA, beginB, endB, state.range(0));

		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
}

static void STL_Par(benchmark::State& state){
	std::vector<OutValType, pad::default_init_allocator<OutValType>> out_mat(state.range(0)*state.range(1));
    std::fill(out_mat.begin(), out_mat.end(), (OutValType)0.0);

	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	auto beginB = out_mat.begin();
	auto endB = out_mat.end();

	for (auto _ : state) {
    	transpose::stl_each_cw(beginA, endA, beginB, state.range(0), std::execution::par);

		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
}

static void STL_Par_Unseq(benchmark::State& state){
	std::vector<OutValType, pad::default_init_allocator<OutValType>> out_mat(state.range(0)*state.range(1));
    std::fill(out_mat.begin(), out_mat.end(), (OutValType)0.0);

	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	auto beginB = out_mat.begin();
	auto endB = out_mat.end();

	for (auto _ : state) {
    	transpose::stl_each_cw(beginA, endA, beginB, state.range(0), std::execution::par_unseq);

		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
}

static void TBB(benchmark::State& state){
	std::vector<OutValType, pad::default_init_allocator<OutValType>> out_mat(state.range(0)*state.range(1));
    std::fill(out_mat.begin(), out_mat.end(), (OutValType)0.0);

	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	auto beginB = out_mat.begin();
	auto endB = out_mat.end();

	for (auto _ : state) {
		transpose::tbb(beginA, beginB, endB, state.range(0), state.range(2), transPart);

		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
}

static void TBB_OMP_SIMD(benchmark::State& state){
	std::vector<OutValType, pad::default_init_allocator<OutValType>> out_mat(state.range(0)*state.range(1));
    std::fill(out_mat.begin(), out_mat.end(), (OutValType)0.0);

	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	auto beginB = out_mat.begin();
	auto endB = out_mat.end();

	for (auto _ : state) {
		transpose::tbbSIMD(beginA, beginB, endB, state.range(0), state.range(2), transPart);
		
		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
}

static void TBB_AVX(benchmark::State& state){
	std::vector<OutValType, pad::default_init_allocator<OutValType>> out_mat(state.range(0)*state.range(1));
    std::fill(out_mat.begin(), out_mat.end(), (OutValType)0.0);

	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	auto beginB = out_mat.begin();
	auto endB = out_mat.end();

	for (auto _ : state) {
		transpose::C_tbbIntrin(beginA, beginB, endB, state.range(0), state.range(2), transPart);
		
		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
}

static void OMP(benchmark::State& state){
	std::vector<OutValType, pad::default_init_allocator<OutValType>> out_mat(state.range(0)*state.range(1));
    std::fill(out_mat.begin(), out_mat.end(), (OutValType)0.0);

	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	auto beginB = out_mat.begin();
	auto endB = out_mat.end();

	for (auto _ : state) {
		transpose::openMP(beginA, beginB, endB, state.range(0));
		
		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
}

static void OMP_Tiled(benchmark::State& state){
	std::vector<OutValType, pad::default_init_allocator<OutValType>> out_mat(state.range(0)*state.range(1));
    std::fill(out_mat.begin(), out_mat.end(), (OutValType)0.0);

	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	auto beginB = out_mat.begin();
	auto endB = out_mat.end();

	for (auto _ : state) {
		transpose::openMPTiled(beginA, beginB, endB, state.range(0), state.range(2));
		
		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
}

static void OMP_Tiled_SIMD(benchmark::State& state){
	std::vector<OutValType, pad::default_init_allocator<OutValType>> out_mat(state.range(0)*state.range(1));
    std::fill(out_mat.begin(), out_mat.end(), (OutValType)0.0);

	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	auto beginB = out_mat.begin();
	auto endB = out_mat.end();

	for (auto _ : state) {
		transpose::openMPSIMD(beginA, beginB, endB, state.range(0), state.range(2));
		
		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
}

static void OMP_Tiled_AVX(benchmark::State& state){
	std::vector<OutValType, pad::default_init_allocator<OutValType>> out_mat(state.range(0)*state.range(1));
    std::fill(out_mat.begin(), out_mat.end(), (OutValType)0.0);

	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	auto beginB = out_mat.begin();
	auto endB = out_mat.end();

	for (auto _ : state) {
		transpose::C_openMPIntrin(beginA, beginB, endB, state.range(0), state.range(2));
		
		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
}

BENCHMARK(Serial)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);// ->Iterations(10);
BENCHMARK(STL_Par)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);// ->Iterations(10);
BENCHMARK(STL_Par_Unseq)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);// ->Iterations(10);

BENCHMARK(TBB)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);// ->Iterations(10);
// BENCHMARK(TBB_OMP_SIMD)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);
BENCHMARK(TBB_AVX)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);
// BENCHMARK(hwLoc)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);

BENCHMARK(OMP)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond)->Iterations(10);
BENCHMARK(OMP_Tiled)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);
// BENCHMARK(OMP_Tiled_SIMD)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);
BENCHMARK(OMP_Tiled_AVX)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);
BENCHMARK_MAIN(); 