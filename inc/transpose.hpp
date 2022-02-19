#pragma once
// #include <array>
// #include <cmath>
// #include <cstdio>
#include <iostream>
// #include <chrono>
// #include <numeric>
#include <algorithm>
#include <oneapi/tbb.h>
#include <iterator>
#include <execution>
#include <x86intrin.h>

using ElemntType = float;

void tran(float* mat, float* matT, size_t in_w, size_t out_w) {
// https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2

  __m256  r0, r1, r2, r3, r4, r5, r6, r7;
  __m256  t0, t1, t2, t3, t4, t5, t6, t7;

  r0 = _mm256_load_ps(&mat[0*in_w]);
  r1 = _mm256_load_ps(&mat[1*in_w]);
  r2 = _mm256_load_ps(&mat[2*in_w]);
  r3 = _mm256_load_ps(&mat[3*in_w]);
  r4 = _mm256_load_ps(&mat[4*in_w]);
  r5 = _mm256_load_ps(&mat[5*in_w]);
  r6 = _mm256_load_ps(&mat[6*in_w]);
  r7 = _mm256_load_ps(&mat[7*in_w]);

  t0 = _mm256_unpacklo_ps(r0, r1);
  t1 = _mm256_unpackhi_ps(r0, r1);
  t2 = _mm256_unpacklo_ps(r2, r3);
  t3 = _mm256_unpackhi_ps(r2, r3);
  t4 = _mm256_unpacklo_ps(r4, r5);
  t5 = _mm256_unpackhi_ps(r4, r5);
  t6 = _mm256_unpacklo_ps(r6, r7);
  t7 = _mm256_unpackhi_ps(r6, r7);

  r0 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(1,0,1,0));  
  r1 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(3,2,3,2));
  r2 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(1,0,1,0));
  r3 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(3,2,3,2));
  r4 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(1,0,1,0));
  r5 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(3,2,3,2));
  r6 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(1,0,1,0));
  r7 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(3,2,3,2));

  t0 = _mm256_permute2f128_ps(r0, r4, 0x20);
  t1 = _mm256_permute2f128_ps(r1, r5, 0x20);
  t2 = _mm256_permute2f128_ps(r2, r6, 0x20);
  t3 = _mm256_permute2f128_ps(r3, r7, 0x20);
  t4 = _mm256_permute2f128_ps(r0, r4, 0x31);
  t5 = _mm256_permute2f128_ps(r1, r5, 0x31);
  t6 = _mm256_permute2f128_ps(r2, r6, 0x31);
  t7 = _mm256_permute2f128_ps(r3, r7, 0x31);

  _mm256_store_ps(&matT[0*out_w], t0);
  _mm256_store_ps(&matT[1*out_w], t1);
  _mm256_store_ps(&matT[2*out_w], t2);
  _mm256_store_ps(&matT[3*out_w], t3);
  _mm256_store_ps(&matT[4*out_w], t4);
  _mm256_store_ps(&matT[5*out_w], t5);
  _mm256_store_ps(&matT[6*out_w], t6);
  _mm256_store_ps(&matT[7*out_w], t7);
}

void tran2(float* mat, float* matT, size_t in_w, size_t out_w) {
    // https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
  __m256  r0, r1, r2, r3, r4, r5, r6, r7;
  __m256  t0, t1, t2, t3, t4, t5, t6, t7;

  r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[0*in_w+0])), _mm_load_ps(&mat[4*in_w+0]), 1);
  r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[1*in_w+0])), _mm_load_ps(&mat[5*in_w+0]), 1);
  r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[2*in_w+0])), _mm_load_ps(&mat[6*in_w+0]), 1);
  r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[3*in_w+0])), _mm_load_ps(&mat[7*in_w+0]), 1);
  r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[0*in_w+4])), _mm_load_ps(&mat[4*in_w+4]), 1);
  r5 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[1*in_w+4])), _mm_load_ps(&mat[5*in_w+4]), 1);
  r6 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[2*in_w+4])), _mm_load_ps(&mat[6*in_w+4]), 1);
  r7 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(&mat[3*in_w+4])), _mm_load_ps(&mat[7*in_w+4]), 1);

  t0 = _mm256_unpacklo_ps(r0,r1);
  t1 = _mm256_unpackhi_ps(r0,r1);
  t2 = _mm256_unpacklo_ps(r2,r3);
  t3 = _mm256_unpackhi_ps(r2,r3);
  t4 = _mm256_unpacklo_ps(r4,r5);
  t5 = _mm256_unpackhi_ps(r4,r5);
  t6 = _mm256_unpacklo_ps(r6,r7);
  t7 = _mm256_unpackhi_ps(r6,r7);

  __m256 v;

  //r0 = _mm256_shuffle_ps(t0,t2, 0x44);
  //r1 = _mm256_shuffle_ps(t0,t2, 0xEE);  
  v = _mm256_shuffle_ps(t0,t2, 0x4E);
  r0 = _mm256_blend_ps(t0, v, 0xCC);
  r1 = _mm256_blend_ps(t2, v, 0x33);

  //r2 = _mm256_shuffle_ps(t1,t3, 0x44);
  //r3 = _mm256_shuffle_ps(t1,t3, 0xEE);
  v = _mm256_shuffle_ps(t1,t3, 0x4E);
  r2 = _mm256_blend_ps(t1, v, 0xCC);
  r3 = _mm256_blend_ps(t3, v, 0x33);

  //r4 = _mm256_shuffle_ps(t4,t6, 0x44);
  //r5 = _mm256_shuffle_ps(t4,t6, 0xEE);
  v = _mm256_shuffle_ps(t4,t6, 0x4E);
  r4 = _mm256_blend_ps(t4, v, 0xCC);
  r5 = _mm256_blend_ps(t6, v, 0x33);

  //r6 = _mm256_shuffle_ps(t5,t7, 0x44);
  //r7 = _mm256_shuffle_ps(t5,t7, 0xEE);
  v = _mm256_shuffle_ps(t5,t7, 0x4E);
  r6 = _mm256_blend_ps(t5, v, 0xCC);
  r7 = _mm256_blend_ps(t7, v, 0x33);

  _mm256_store_ps(&matT[0*out_w], r0);
  _mm256_store_ps(&matT[1*out_w], r1);
  _mm256_store_ps(&matT[2*out_w], r2);
  _mm256_store_ps(&matT[3*out_w], r3);
  _mm256_store_ps(&matT[4*out_w], r4);
  _mm256_store_ps(&matT[5*out_w], r5);
  _mm256_store_ps(&matT[6*out_w], r6);
  _mm256_store_ps(&matT[7*out_w], r7);
}

// std::iterator_traits<Iter>::value_type

namespace transpose{

// Serial implementation contiguous write
template <class InIter, class OutIter>
void transposeSerial(InIter inStart, OutIter outStart, OutIter outEnd, const size_t in_rows){
    const size_t n_elem = std::distance(outStart, outEnd); 
    const size_t in_columns = n_elem / in_rows;
    for(size_t x = 0; x < in_columns; ++x){
        for(size_t y = 0; y < in_rows; ++y){
            *(outStart + (x * in_rows + y)) = *(inStart + (y * in_columns + x));
        }
    }
}

// Serial verify
template <class InIter, class CmpIter>
void verifyT(InIter inStart, CmpIter outStart, CmpIter outEnd, const size_t in_rows){
    const size_t n_elem = std::distance(outStart, outEnd); 
    const size_t in_columns = n_elem / in_rows;
    for(size_t x = 0; x < in_columns; ++x){
        for(size_t y = 0; y < in_rows; ++y){
            if(*(outStart + (x * in_rows + y)) != *(inStart + (y * in_columns + x))){
                std::cout << "row=" << y << ", column=" << x
                          << ", expected=" << *(inStart + (y * in_columns + x))
                          << ", actual=" << *(outStart + (x * in_rows + y)) << std::endl;
                throw std::runtime_error("wrong results");
            }
        }
    }
}

// doesn't work in parallel!
template <class InIter, class OutIter, class ExecPolicy>
void stl_like(InIter inStart, InIter inEnd, OutIter outStart, const size_t in_rows, ExecPolicy policy){
    const size_t in_columns = std::distance(inStart, inEnd) / in_rows;
    size_t i = 0;
    using InElemType = typename std::iterator_traits<InIter>::value_type;
    std::for_each(policy, inStart, inEnd, [&](InElemType &inElem){
        *outStart = inElem;     
        i++;
        if(i % in_columns == 0)
        {
            std::advance(outStart, -(in_columns-1)*in_rows);  // jump back into first row
            ++outStart;                                       // incrase column by 1
        }
        else
        {
            std::advance(outStart, in_rows);        // step down a row in the Output Matrix
        }
    });
}

// doesn't work in parallel!
template <class InIter, class OutIter>
void stl_likeCoalescedWrite(InIter inStart, OutIter outStart, OutIter outEnd, const size_t in_rows){
    const size_t in_columns = std::distance(outStart, outEnd) / in_rows;
    size_t i = 0;
    using OutElemType = typename std::iterator_traits<OutIter>::value_type;
    std::for_each(outStart, outEnd, [&](OutElemType &outElem){
        outElem = *inStart;
        i++;
        if(!(i % in_rows))
        {
            std::advance(inStart, -(in_rows-1)*in_columns);
            ++inStart;
        }
        else
        {
            std::advance(inStart, in_columns);
        }
        
    });
}

template<typename type>
void obliviousTranspose(int N, int ib, int ie, int jb, int je, type *a, type *b, int gs) {
  int ilen = ie-ib;
  int jlen = je-jb;
  if (ilen > gs ||  jlen > gs) {
     if ( ilen > jlen ) {
       int imid = (ib+ie)/2;
       obliviousTranspose(N, ib, imid, jb, je, a, b, gs);
       obliviousTranspose(N, imid, ie, jb, je, a, b, gs);
     } else {
       int jmid = (jb+je)/2;
       obliviousTranspose(N, ib, ie, jb, jmid, a, b, gs);
       obliviousTranspose(N, ib, ie, jmid, je, a, b, gs);
     }
  } else {
    for (int i = ib; i < ie; ++i) {
      for (int j = jb; j < je; ++j) {
        b[j*N+i] = a[i*N+j];
      }
    }
  }
}

template <class InIter, class OutIter, class Partitioner>
void tbb(InIter inStart, OutIter outStart, OutIter outEnd, const size_t in_rows, const size_t gs, Partitioner& part){
    const size_t n_elem = std::distance(outStart, outEnd); 
    const size_t in_columns = n_elem / in_rows;
    
    oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range2d<size_t>(0, in_rows, gs, 0, in_columns, gs),
            [&](oneapi::tbb::blocked_range2d<size_t> r){
        // transpose subblock of range2d
        for(size_t x = r.cols().begin(); x < r.cols().end(); ++x){
            for(size_t y = r.rows().begin(); y < r.rows().end(); ++y){
                *(outStart + (x * in_rows + y)) = *(inStart + (y * in_columns + x));
            }
        }
    },part
    );
}

template <class inIter, class outIter1, class outIter2, class partitioner>
void tbbSIMD(inIter inStart, outIter1 startIter, outIter2 endIter, size_t in_rows, size_t gs, partitioner& part){
    const size_t n_elem = std::distance(startIter, endIter); 
    const size_t in_columns = n_elem / in_rows;

    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, in_rows, gs, 0, in_columns, gs), [&](tbb::blocked_range2d<size_t> r){
        // transpose subblock of range2d
        for(size_t x = r.cols().begin(); x < r.cols().end(); ++x){
            #pragma omp simd
            for(size_t y = r.rows().begin(); y < r.rows().end(); ++y){
                *(startIter + (x * in_rows + y)) = *(inStart + (y * in_columns + x));
            }
        }
    },part
    );
}

template <class inIter, class outIter1, class outIter2, class partitioner>
void tbbIntrin(inIter inStart, outIter1 startIter, outIter2 endIter, size_t in_rows, size_t gs, partitioner& part){
    const size_t n_elem = std::distance(startIter, endIter); 
    const size_t in_columns = n_elem / in_rows;

    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, in_rows, gs, 0, in_columns, gs), [&](tbb::blocked_range2d<size_t> r){
        size_t distRows = r.rows().end() - r.rows().begin();
        size_t distCols = r.cols().end() - r.cols().begin();
        size_t restRows = distRows % 8;
        size_t restCols = distCols % 8;
        auto restStartR = r.rows().end() - restRows;
        auto restStartC = r.cols().end() - restCols;

        // transpose regular 8x8 blocks
        for(size_t x = r.cols().begin(); x + 8 <= r.cols().end(); x+=8){
            for(size_t y = r.rows().begin(); y + 8 <= r.rows().end(); y+=8){            // y+8 < rows.end beacause we transpose 8x8 block in one iteration
                tran(&(*(inStart + (x + y * in_columns))), &(*(startIter + (y + x * in_rows))), in_columns, in_rows);
            }
        }

        // transpose rest at North East
        for(size_t x = restStartC; x < r.cols().end(); ++x){
            for(size_t y = 0; y < r.rows().end(); ++y){
                *(startIter + (x * in_rows + y)) = *(inStart + (y * in_columns + x));
            }
        }

        // transpose rest at South West
        for(size_t x = 0; x < restStartC; ++x){
            for(size_t y = restStartR; y < r.rows().end(); ++y){
                *(startIter + (x * in_rows + y)) = *(inStart + (y * in_columns + x));
            }
        }
    },part
    );
    
}

template <class inIter, class outIter1, class outIter2, class partitioner>
void tbb_coal_r(inIter inStart, outIter1 startIter, outIter2 endIter, size_t in_rows, size_t gs, partitioner& part){
    const size_t n_elem = std::distance(startIter, endIter); 
    const size_t in_columns = n_elem / in_rows;

    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, in_rows, gs, 0, in_columns, gs), [&](tbb::blocked_range2d<size_t> r){
        // transpose subblock of range2d
        for(size_t y = r.rows().begin(); y < r.rows().end(); ++y){
            for(size_t x = r.cols().begin(); x < r.cols().end(); ++x){
                *(startIter + (x * in_rows + y)) = *(inStart + (y * in_columns + x));
            }
        }
    },part
    );
}

template <class inIter, class outIter1, class outIter2>
void openMP(inIter inStart, outIter1 startIter, outIter2 endIter, const size_t in_rows){
    const size_t n_elem = std::distance(startIter, endIter); 
    const size_t in_columns = n_elem / in_rows;

    #pragma omp parallel for schedule(static) /**collapse(2)**/
    for(size_t y = 0; y < in_columns; y++){
        for(size_t x = 0; x < in_rows; ++x){
            *(startIter + (y * in_rows + x)) = *(inStart + (x * in_columns + y));
        }
    }
}

template <class inIter, class outIter1, class outIter2>
void openMPTiled(inIter inStart, outIter1 startIter, outIter2 endIter, const size_t in_rows, const size_t blocksize){
    const size_t n_elem = std::distance(startIter, endIter); 
    const size_t in_columns = n_elem / in_rows;
    
    // calculate regular blocksize x blocksize blocks
    #pragma omp parallel for schedule(static)
    for (size_t i = blocksize; i <= in_columns; i += blocksize) {                   // äußere Schleife von links nach rechts
        for (size_t j = blocksize; j <= in_rows; j += blocksize) {           // innere Schleife von oben nach unten durch matrix
            // transpose the block beginning at [i,j]
            for (size_t k = i - blocksize; k < i; ++k) {         //äußere Schleife von links nach rechts durch Block
                for (size_t l = j - blocksize; l < j; ++l) {        //innere Schleife von oben nach unten durch Block
                    *(startIter + (l + k * in_rows)) = *(inStart + (k + l * in_columns));
                }
            }
        }
    }

    size_t restCols = in_columns % blocksize;
    size_t restRows = in_rows % blocksize;

    size_t restStartC = in_columns - restCols;
    size_t restStartR = in_rows - restRows;

    // calculate rest at North East (input)
    #pragma omp parallel for schedule(static)
    for (size_t i = restStartC; i < in_columns; ++i) {                   
        for (size_t j = 0; j < in_rows; ++j) {           
            *(startIter + (j + i * in_rows)) = *(inStart + (i + j * in_columns));
        }
    }

    // calculate rest at South West (input)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < restStartC; ++i) {                   
        for (size_t j = restStartR; j < in_rows; ++j) {          
            *(startIter + (j + i * in_rows)) = *(inStart + (i + j * in_columns));
        }
    }
}

template <class inIter, class outIter1, class outIter2>
void openMPSIMD(inIter inStart, outIter1 startIter, outIter2 endIter, const size_t in_rows, const size_t blocksize){
    const size_t n_elem = std::distance(startIter, endIter); 
    const size_t in_columns = n_elem / in_rows;
    
    #pragma omp parallel for schedule(static)
    for (size_t i = blocksize; i <= in_columns; i += blocksize) {            // äußere Schleife von links nach rechts
        for (size_t j = blocksize; j <= in_rows; j += blocksize) {           // innere Schleife von oben nach unten durch matrix
            // transpose the block beginning at [i,j]
            for (size_t k = i-blocksize; k < i; ++k) {         //äußere Schleife von links nach rechts durch Block
                #pragma omp simd            
                for (size_t l = j-blocksize; l < j; ++l) {        //innere Schleife von oben nach unten durch Block
                    *(startIter + (l + k * in_rows)) = *(inStart + (k + l * in_columns));
                }
            }
        }
    }

    size_t restCols = in_columns % blocksize;
    size_t restRows = in_rows % blocksize;

    size_t restStartC = in_columns - restCols;
    size_t restStartR = in_rows - restRows;

    // calculate rest at North East (input)
    #pragma omp parallel for schedule(static)
    for (size_t i = restStartC; i < in_columns; ++i) {                   
        for (size_t j = 0; j < in_rows; ++j) {           
            *(startIter + (j + i * in_rows)) = *(inStart + (i + j * in_columns));
        }
    }

    // calculate rest at South West (input)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < restStartC; ++i) {                   
        for (size_t j = restStartR; j < in_rows; ++j) {          
            *(startIter + (j + i * in_rows)) = *(inStart + (i + j * in_columns));
        }
    }
}

template <ssize_t simd_width, class inIter, class outIter1, class outIter2>
void openMPIntrin(inIter inStart, outIter1 startIter, outIter2 endIter, const size_t in_rows, const size_t blocksize){
    //blocksize can only be a multiple of 8
    const size_t n_elem = std::distance(startIter, endIter); 
    const size_t in_columns = n_elem / in_rows;
    
    #pragma omp parallel for schedule(static)
    for (size_t i = blocksize; i <= in_columns; i += blocksize) {                   // äußere Schleife von links nach rechts
        for (size_t j = blocksize; j <= in_rows; j += blocksize) {           // innere Schleife von oben nach unten durch matrix
            for (size_t k = i-blocksize; k < i; k+=8)
            {
                for (size_t l = j-blocksize; l < j; l+=8)
                {
                    tran(&(*(inStart + (k + l * in_columns))), &(*(startIter + (l + k * in_rows))), in_columns, in_rows);
                }
                
            }           
        }
    }

    size_t restCols = in_columns % blocksize;
    size_t restRows = in_rows % blocksize;

    size_t restStartC = in_columns - restCols;
    size_t restStartR = in_rows - restRows;

    // calculate rest at North East (input)
    #pragma omp parallel for schedule(static)
    for (size_t i = restStartC; i < in_columns; ++i) {                   
        for (size_t j = 0; j < in_rows; ++j) {           
            *(startIter + (j + i * in_rows)) = *(inStart + (i + j * in_columns));
        }
    }

    // calculate rest at South West (input)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < restStartC; ++i) {                   
        for (size_t j = restStartR; j < in_rows; ++j) {          
            *(startIter + (j + i * in_rows)) = *(inStart + (i + j * in_columns));
        }
    }
}



template <class Iter1, class Iter2, class OutIter>
void simple_cp(Iter1 inBegin, Iter2 inEnd, OutIter outBegin){
    int len = std::distance(inBegin, inEnd);
    #pragma omp parallel for schedule(static)
	for (size_t i = 0; i < len; ++i)
	{
		*(outBegin + i) = *(inBegin + i);
	}
}

template <class Iter1, class Iter2, class OutIter>
void matrix_cp(Iter1 inBegin, Iter2 inEnd, OutIter outBegin, size_t in_rows, char flag){
    const size_t n_elem = std::distance(inBegin, inEnd);
    const size_t in_columns = n_elem / in_rows;
    const size_t out_columns = in_rows;

    switch (flag)
    {
    case 'S': // copy Matrix with row-stride
        #pragma omp parallel for schedule(static)
        for(size_t x = 0; x < in_columns; ++x){
            for(size_t y = 0; y < in_rows; ++y){
                *(outBegin + (y * in_columns + x)) = *(inBegin + (y * in_columns + x));
            }
        }
        break;

    case 'C': // copy Matrix
        #pragma omp parallel for schedule(static)
        for(size_t y = 0; y < in_rows; ++y){
            for(size_t x = 0; x < in_columns; ++x){
                *(outBegin + (y * in_columns + x)) = *(inBegin + (y * in_columns + x));
            }
        }
        break;

    case 'T': // transpose coalescing read Matrix
        #pragma omp parallel for schedule(static)
        for(size_t y = 0; y < in_rows; ++y){
            for(size_t x = 0; x < in_columns; ++x){
                *(outBegin + (x * out_columns + y)) = *(inBegin + (y * in_columns + x));
            }
        }
        break;

    case 't': // transpose coalescing write Matrix
        #pragma omp parallel for schedule(static)
        for(size_t x = 0; x < in_columns; ++x){
            for(size_t y = 0; y < in_rows; ++y){
                *(outBegin + (x * out_columns + y)) = *(inBegin + (y * in_columns + x));
            }
        }
        break;
    
    default:
        break;
    }

}

}