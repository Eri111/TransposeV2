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

#include "dataV2.hpp"
#include "transpose.hpp"
// #include "pad/pinningobserver.hpp"


// constexpr size_t simd_width = 32;
constexpr bool useSerInit = false;
constexpr bool do_verify = true;


// bool floatEquals(double lhs, double rhs, double epsilon = 1e-5) {
//     return std::abs(lhs - rhs) < epsilon;
// }

static void BenchmarkArguments(benchmark::internal::Benchmark* b) {
	const ssize_t lowerLimit = 15;
	const ssize_t upperLimit = 15;

	const ssize_t lowerGS = 64;
	const ssize_t upperGS = 64;
	for (auto j = lowerGS; j <= upperGS; j += 4)
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

static void TBB(benchmark::State& state){
	pad::arrayDataV2<InValType> *data = nullptr;

	if(useSerInit){
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1), state.range(2), "TBB", dataPart);
	}
	
	auto iterators = data->get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = data->get_ptr();

	for (auto _ : state) {
		transpose::tbb(beginA, beginB, endB, state.range(0), state.range(2), transPart);

		benchmark::DoNotOptimize(dataB);
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void TBB_OMP_SIMD(benchmark::State& state){
	pad::arrayDataV2<InValType> *data = nullptr;

	if(useSerInit){
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1), state.range(2), "TBB", dataPart);
	}
	
	auto iterators = data->get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = data->get_ptr();

	for (auto _ : state) {
		transpose::tbbSIMD(beginA, beginB, endB, state.range(0), state.range(2), transPart);
		
		benchmark::DoNotOptimize(dataB);
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void TBB_AVX(benchmark::State& state){
	pad::arrayDataV2<InValType> *data = nullptr;

	if(useSerInit){
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1), state.range(2), "TBB", dataPart);
	}
	
	auto iterators = data->get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = data->get_ptr();

	for (auto _ : state) {
		transpose::tbbIntrin(beginA, beginB, endB, state.range(0), state.range(2), transPart);
		
		benchmark::DoNotOptimize(dataB);
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void OMP(benchmark::State& state){
	pad::arrayDataV2<InValType> *data = nullptr;

	if(useSerInit){
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1), state.range(2), "OMP");
	}
	
	auto iterators = data->get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = data->get_ptr();

	for (auto _ : state) {
		transpose::openMP(beginA, beginB, endB, state.range(0));
		
		benchmark::DoNotOptimize(dataB);
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void OMP_Tiled(benchmark::State& state){
	pad::arrayDataV2<InValType> *data = nullptr;

	if(useSerInit){
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1), state.range(2), "OMP");
	}
	
	auto iterators = data->get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = data->get_ptr();

	for (auto _ : state) {
		transpose::openMPTiled(beginA, beginB, endB, state.range(0), state.range(2));
		
		benchmark::DoNotOptimize(dataB);
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void OMP_Tiled_SIMD(benchmark::State& state){
	pad::arrayDataV2<InValType> *data = nullptr;

	if(useSerInit){
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1), state.range(2), "OMP");
	}
	
	auto iterators = data->get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = data->get_ptr();

	for (auto _ : state) {
		transpose::openMPSIMD(beginA, beginB, endB, state.range(0), state.range(2));
		
		benchmark::DoNotOptimize(dataB);
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void OMP_Tiled_AVX(benchmark::State& state){
	pad::arrayDataV2<InValType> *data = nullptr;

	if(useSerInit){
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1), state.range(2), "OMP");
	}
	
	auto iterators = data->get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = data->get_ptr();

	for (auto _ : state) {
		transpose::openMPIntrin(beginA, beginB, endB, state.range(0), state.range(2));
		
		benchmark::DoNotOptimize(dataB);
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}


BENCHMARK(TBB)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond)->Iterations(10);
BENCHMARK(TBB_OMP_SIMD)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond)->Iterations(10);
BENCHMARK(TBB_AVX)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond)->Iterations(10);

BENCHMARK(OMP)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond)->Iterations(10);
BENCHMARK(OMP_Tiled)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond)->Iterations(10);
BENCHMARK(OMP_Tiled_SIMD)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond)->Iterations(10);
BENCHMARK(OMP_Tiled_AVX)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond)->Iterations(10);

BENCHMARK_MAIN();