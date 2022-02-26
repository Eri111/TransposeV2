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
#include "iterator.hpp"

static const size_t in_rows = 13;
static const size_t in_cols = 14;
static const size_t tile_width = 8;

template <typename T>
void printMat(T mat, size_t in_rows, size_t in_cols){
    size_t out_rows = in_cols;
    size_t out_cols = in_rows;
    for (size_t y = 0; y < out_rows; y++)
    {
        for (size_t x = 0; x < out_cols; x++)
        {
            std::cout << std::setw(4) << mat[x + y*out_cols] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char const *argv[])
{
    // pad::arrayDataV2<InValType> dataS(4, 3);
    // dataS.printA();
    // dataS.printB();

    pad::arrayDataV2<InValType> dataOMP(in_rows, in_cols, tile_width, "OMP");
    dataOMP.printA();
    

    // pad::arrayDataV2<InValType> dataTBB(in_rows, in_cols, tile_width, "TBB", dataPart);
    // dataTBB.printA();
    // dataTBB.printB();
    

    auto iterators = dataOMP.get_range();
    auto [beginA, endA] = std::get<0>(iterators);
    auto [beginB, endB] = std::get<1>(iterators);
    auto [dataA, dataB] = dataOMP.get_ptr();

    // transpose::transposeSerial(beginA, beginB, endB, in_rows);
    // std::execution::par_unseq,
    // transpose::stl_like(beginA, endA, beginB, in_rows, std::execution::seq);
    // transpose::stl_like(beginA, endA, beginB, in_rows, std::execution::par);
    // transpose::stl_like(beginA, endA, beginB, in_rows, std::execution::par_unseq);
    // transpose::stl_likeCoalescedWrite(beginA, beginB, endB, in_rows);
    // transpose::tbb(beginA, beginB, endB, in_rows, tile_width, transPart);
    // transpose::tbbSIMD(beginA, beginB, endB, in_rows, tile_width, transPart);
    // transpose::tbbIntrin(beginA, beginB, endB, in_rows, tile_width, transPart);
    // transpose::tbb_coal_r(beginA, beginB, endB, in_rows, tile_width, transPart);

    // transpose::openMP(beginA, beginB, endB, in_rows);
    // transpose::openMPTiled(beginA, beginB, endB, in_rows, tile_width);
    // transpose::openMPSIMD(beginA, beginB, endB, in_rows, tile_width);
    // transpose::openMPIntrin(beginA, beginB, endB, in_rows, tile_width);

    // transpose::stl_each_cr(beginA, endA, beginB, in_rows, std::execution::par_unseq);
    // transpose::stl_each_cw(beginA, endA, beginB, in_rows, std::execution::par_unseq);
    
    std::vector<float, pad::default_init_allocator<float>> out_mat(in_rows*in_cols);
    std::fill(out_mat.begin(), out_mat.end(), 0);
	pad::Arith_Iterator beginIter(0, [](ssize_t idx) { return (InValType)idx; });

    // transpose::tbb(beginIter, out_mat.begin(), out_mat.end(), in_rows, tile_width, transPart);
    transpose::tbbSIMD(beginIter, out_mat.begin(), out_mat.end(), in_rows, tile_width, transPart);
    // transpose::openMPSIMD(beginIter, out_mat.begin(), out_mat.end(), in_rows, tile_width);
    // transpose::C_openMPIntrin(beginIter, out_mat.begin(), out_mat.end(), in_rows, 8);

    printMat(out_mat, in_rows, in_cols);

    // dataOMP.printB();
    // transpose::verifyT(beginA, beginB, endB, in_rows);
    // transpose::verifyPar(beginA, beginB, endB, in_rows);

    // std::cout << "Hello World";
    return 0;
}
