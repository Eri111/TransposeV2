#include <benchmark/benchmark.h>
#include <iostream>
#include <oneapi/tbb.h>

#define DATA_POLICY dynamic
#define TRANSPOSE_POLICY dynamic

using DataPartitioner = oneapi::tbb::simple_partitioner;
static DataPartitioner dataPart;

using TransposePartitioner = oneapi::tbb::simple_partitioner;
static TransposePartitioner transPart;

using InValType = float;
using OutValType = float;

#include "dataV2.hpp"
#include "transpose.hpp"
#include "pinningobserver.hpp"


// constexpr size_t simd_width = 32;
constexpr bool useSerInit = true;
constexpr bool do_verify = false;
constexpr size_t avx_gs = 128;

static void BenchmarkArguments(benchmark::internal::Benchmark* b) {
	const ssize_t lowerLimit = 5;
	const ssize_t upperLimit = 15;

	const ssize_t lowerGS = 64;
	const ssize_t upperGS = 64;
	for (auto j = lowerGS; j <= upperGS; j += 8)
	{	
		for (auto i = lowerLimit; i <= upperLimit; ++i)
		{
			b->Args({(1 << i), (1 << i), (j)}); // first arg in_rows (Zeilen ANZAHL), second arg in_columns
		}
	}
}

void transposeCustomCounter(benchmark::State& state) {
	state.counters["Elements"] = state.range(0) * state.range(1);
	state.counters["Bytes_processed"] = state.range(1) * state.range(0) * sizeof(InValType);
	state.counters["Matrix_height"] = (int)state.range(0);
	state.counters["Matrix_width"] = (int)state.range(1);
	state.counters["Tile_width"] = (int)state.range(2);
}

static void Serial(benchmark::State& state){
	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));

	pad::arrayDataV2<OutValType> *data = nullptr;
	if(useSerInit){
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1));
	}
	auto iterators = data->get_range();
	auto [beginB, endB] = std::get<1>(iterators);

	for (auto _ : state) {
		transpose::transposeSerial(beginA, beginB, endB, state.range(0));
		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void STL_Par(benchmark::State& state){
	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	pad::arrayDataV2<OutValType> *data = nullptr;
	if(useSerInit){
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1), state.range(2), "TBB", dataPart);
	}
	auto iterators = data->get_range();
	auto [beginB, endB] = std::get<1>(iterators);

	for (auto _ : state) {
    	transpose::stl_each_cw(beginA, endA, beginB, state.range(0), std::execution::par);
		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void STL_Par_Unseq(benchmark::State& state){
	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	pad::arrayDataV2<OutValType> *data = nullptr;
	if(useSerInit){
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1), state.range(2), "TBB", dataPart);
	}
	auto iterators = data->get_range();
	auto [beginB, endB] = std::get<1>(iterators);

	for (auto _ : state) {
    	transpose::stl_each_cw(beginA, endA, beginB, state.range(0), std::execution::par_unseq);
		
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void TBB(benchmark::State& state){
	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	pad::arrayDataV2<OutValType> *data = nullptr;
	if(useSerInit){
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1), state.range(2), "TBB", dataPart);
	}
	auto iterators = data->get_range();
	auto [beginB, endB] = std::get<1>(iterators);

	for (auto _ : state) {
		transpose::tbb(beginA, beginB, endB, state.range(0), state.range(2), transPart);
	
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void TBB_OMP_SIMD(benchmark::State& state){
	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	pad::arrayDataV2<OutValType> *data = nullptr;
	if(useSerInit){
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1), state.range(2), "TBB", dataPart);
	}
	auto iterators = data->get_range();
	auto [beginB, endB] = std::get<1>(iterators);

	for (auto _ : state) {
		transpose::tbbSIMD(beginA, beginB, endB, state.range(0), state.range(2), transPart);
				
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void TBB_AVX(benchmark::State& state){
	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	pad::arrayDataV2<OutValType> *data = nullptr;
	if(useSerInit){
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1), avx_gs, "TBB", dataPart);
	}
	auto iterators = data->get_range();
	auto [beginB, endB] = std::get<1>(iterators);

	for (auto _ : state) {
		transpose::C_tbbIntrin(beginA, beginB, endB, state.range(0), avx_gs, transPart);
			
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void OMP(benchmark::State& state){
	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	pad::arrayDataV2<OutValType> *data = nullptr;
	if(useSerInit){
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1), state.range(2), "OMP");
	}
	auto iterators = data->get_range();
	auto [beginB, endB] = std::get<1>(iterators);

	for (auto _ : state) {
		transpose::openMP(beginA, beginB, endB, state.range(0));
				
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void OMP_Tiled(benchmark::State& state){
	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	pad::arrayDataV2<OutValType> *data = nullptr;
	if(useSerInit){
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1), state.range(2), "OMP");
	}
	auto iterators = data->get_range();
	auto [beginB, endB] = std::get<1>(iterators);

	for (auto _ : state) {
		transpose::openMPTiled(beginA, beginB, endB, state.range(0), state.range(2));
				
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void OMP_Tiled_SIMD(benchmark::State& state){
	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	pad::arrayDataV2<OutValType> *data = nullptr;
	if(useSerInit){
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1), state.range(2), "OMP");
	}
	auto iterators = data->get_range();
	auto [beginB, endB] = std::get<1>(iterators);

	for (auto _ : state) {
		transpose::openMPSIMD(beginA, beginB, endB, state.range(0), state.range(2));
			
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void OMP_Tiled_AVX(benchmark::State& state){
	pad::Arith_Iterator beginA(0, [](ssize_t idx) { return (InValType)idx; });
	pad::Arith_Iterator endA = beginA + (state.range(0)*state.range(1));
	
	pad::arrayDataV2<OutValType> *data = nullptr;
	if(useSerInit){
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<OutValType>(state.range(0), state.range(1), avx_gs, "OMP");
	}
	auto iterators = data->get_range();
	auto [beginB, endB] = std::get<1>(iterators);

	for (auto _ : state) {
		transpose::C_openMPIntrin(beginA, beginB, endB, state.range(0), avx_gs);
				
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

BENCHMARK(Serial)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(STL_Par)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(STL_Par_Unseq)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);

BENCHMARK(TBB)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(TBB_OMP_SIMD)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); 
BENCHMARK(TBB_AVX)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); 
BENCHMARK(OMP)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); 
BENCHMARK(OMP_Tiled)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); 
BENCHMARK(OMP_Tiled_SIMD)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); 
BENCHMARK(OMP_Tiled_AVX)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); 
BENCHMARK_MAIN(); 