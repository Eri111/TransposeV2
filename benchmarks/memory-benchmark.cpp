#include <benchmark/benchmark.h>
#include <iostream>
#include <oneapi/tbb.h>

#define DATA_POLICY static
#define TRANSPOSE_POLICY static

using DataPartitioner = oneapi::tbb::static_partitioner;
static DataPartitioner dataPart;

using TransposePartitioner = oneapi::tbb::static_partitioner;
static TransposePartitioner transPart;

using InValType = float;

#include "dataV2.hpp"
#include "transpose.hpp"
#include "pinningobserver.hpp"


// constexpr size_t simd_width = 32;
constexpr bool useSerInit = false;
constexpr bool do_verify = false;

constexpr size_t avx_gs = 64;


// bool floatEquals(double lhs, double rhs, double epsilon = 1e-5) {
//     return std::abs(lhs - rhs) < epsilon;
// }

static void BenchmarkArguments(benchmark::internal::Benchmark* b) {
	const ssize_t lowerLimit = 5;
	const ssize_t upperLimit = 15;

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
	pad::arrayDataV2<InValType> *data = nullptr;

	if(useSerInit){
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1));
	}
	else{
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1));
	}
	
	auto iterators = data->get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = data->get_ptr();

	for (auto _ : state) {
		transpose::transposeSerial(beginA, beginB, endB, state.range(0));

		benchmark::DoNotOptimize(dataB);
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void STL_Par(benchmark::State& state){
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
    	transpose::stl_each_cw(beginA, endA, beginB, state.range(0), std::execution::par);

		benchmark::DoNotOptimize(dataB);
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void STL_Par_Unseq(benchmark::State& state){
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
    	transpose::stl_each_cw(beginA, endA, beginB, state.range(0), std::execution::par_unseq);

		benchmark::DoNotOptimize(dataB);
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
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
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1), avx_gs, "TBB", dataPart);
	}
	
	auto iterators = data->get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = data->get_ptr();

	for (auto _ : state) {
		transpose::tbbIntrin(beginA, beginB, endB, state.range(0), avx_gs, transPart);
		
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
		data = new pad::arrayDataV2<InValType>(state.range(0), state.range(1), avx_gs, "OMP");
	}
	
	auto iterators = data->get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = data->get_ptr();

	for (auto _ : state) {
		transpose::openMPIntrin(beginA, beginB, endB, state.range(0), avx_gs);
		
		benchmark::DoNotOptimize(dataB);
		benchmark::ClobberMemory();
	}
	if(do_verify){
		transpose::verifyPar(beginA, beginB, endB, state.range(0));
	}
	transposeCustomCounter(state);
	delete data;
}

static void hwLoc(benchmark::State& state){

	int thds_per_node = 32;
	size_t size = (size_t) (state.range(0) * state.range(1)) / 4;

	float **data_in = new float*[4];
	float **data_out = new float*[4];

	hwloc_topology_t topo;
	hwloc_topology_init(&topo);
	hwloc_topology_load(topo);

	numa::alloc_mem_per_node(topo, data_in, data_out, size, state.range(0));
	for (auto _ : state){
		numa::alloc_thr_per_node(topo, data_in, data_out, size, thds_per_node, state.range(0), state.range(2));
		benchmark::DoNotOptimize(data_out);
		benchmark::ClobberMemory();
	}
	transposeCustomCounter(state);

	for (int i = 0; i < 4; i++){
		hwloc_free(topo, data_in[i], size);
		hwloc_free(topo, data_out[i], size);
	}
	hwloc_topology_destroy(topo);
	delete [] data_in;
	delete [] data_out;
}

// BENCHMARK(Serial)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);// ->Iterations(10);
// BENCHMARK(STL_Par)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);// ->Iterations(10);
// BENCHMARK(STL_Par_Unseq)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);// ->Iterations(10);

BENCHMARK(TBB)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond);// ->Iterations(10);
BENCHMARK(TBB_OMP_SIMD)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);
BENCHMARK(TBB_AVX)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);
// BENCHMARK(hwLoc)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);

BENCHMARK(OMP)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);
BENCHMARK(OMP_Tiled)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);
BENCHMARK(OMP_Tiled_SIMD)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);
BENCHMARK(OMP_Tiled_AVX)->Apply(BenchmarkArguments)->UseRealTime()->Unit(benchmark::kMicrosecond); //->Iterations(10);

BENCHMARK_MAIN();