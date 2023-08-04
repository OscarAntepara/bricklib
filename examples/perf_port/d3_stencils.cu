#include "stencils/stencils_cu.h"
#include "d3_stencils.h"
#include <iostream>
#include <ctime>
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include "stencils/cudavfold.h"

#include "macros_coeffs.h"

#define VSVEC "CUDA"
#define WARPSIZE 32
#undef VFOLD
#define VFOLD 1, 32

#undef TILE
#define TILE 4

#undef PADDING
#define PADDING TILE

#undef GZ
#define GZ TILE

#define TILEX 32
#define GZX TILEX
#define PADDINGX TILEX

#undef N 
#define N 512

#define NX N / 1 //4
#define NY N / 1 //4
#define NZ N * 1 //16

#define STRIDEX (NX + 2 * (GZX + PADDINGX))
#define STRIDEY (NY + 2 * (GZ + PADDING))
#define STRIDEZ (NZ + 2 * (GZ + PADDING))

#define STRIDEGX (NX + 2 * GZX)
#define STRIDEGY (NY + 2 * GZ)
#define STRIDEGZ (NZ + 2 * GZ)

#define STRIDEBX ((NX + 2 * GZX) / TILEX)
#define STRIDEBY ((NY + 2 * GZ) / TILE)
#define STRIDEBZ ((NZ + 2 * GZ) / TILE)

#define NBX (NX / TILEX)
#define NBY (NY / TILE)
#define NBZ (NZ / TILE)

#define GBX (GZX / TILEX)
#undef BDIM
#define BDIM TILE,TILE,TILEX

#undef _TILEFOR
#define _TILEFOR _Pragma("omp parallel for collapse(2)") \
for (long tk = PADDING; tk < PADDING + STRIDEGZ; tk += TILE) \
for (long tj = PADDING; tj < PADDING + STRIDEGY; tj += TILE) \
for (long ti = PADDINGX; ti < PADDINGX + STRIDEGX; ti += TILEX) \
for (long k = tk; k < tk + TILE; ++k) \
for (long j = tj; j < tj + TILE; ++j) \
_Pragma("omp simd") \
for (long i = ti; i < ti + TILEX; ++i)

__global__ void
d3star_brick(unsigned (*grid)[STRIDEBY][STRIDEBX], Brick <Dim<BDIM>, Dim<VFOLD>> bIn, Brick <Dim<BDIM>, Dim<VFOLD>> bOut){
            //bElem *coeff) {
  long tj = GB + blockIdx.y;
  long tk = GB + blockIdx.z;
  long ti = GBX + blockIdx.x;
  long k = threadIdx.z;
  long j = threadIdx.y;
  long i = threadIdx.x;
  unsigned b = grid[tk][tj][ti];
  ST_STAR_BRICK_GPU;
  /*
  bOut[b][k][j][i] = coeff[0] * bIn[b][k][j][i];
  #pragma unroll
  for (int a = 1; a <= STAR_STENCIL_RADIUS; a++) {
      bOut[b][k][j][i] += coeff[a] * (
          bIn[b][k][j][i + a] + bIn[b][k][j + a][i] + bIn[b][k + a][j][i] +
          bIn[b][k][j][i - a] + bIn[b][k][j - a][i] + bIn[b][k - a][j][i]
      );
  }
  */
}

__global__ void
d3cube_brick(unsigned (*grid)[STRIDEBY][STRIDEBX], Brick <Dim<BDIM>, Dim<VFOLD>> bIn, Brick <Dim<BDIM>, Dim<VFOLD>> bOut){
            //bElem (*coeff)[8][8]) {
  long tj = GB + blockIdx.y;
  long tk = GB + blockIdx.z;
  long ti = GBX + blockIdx.x;
  long k = threadIdx.z;
  long j = threadIdx.y;
  long i = threadIdx.x;
  unsigned b = grid[tk][tj][ti];
  ST_CUBE_BRICK_GPU;
  /*
  bOut[b][k][j][i] = 0.0;
  #pragma unroll
  for (int k_diff = -CUBE_STENCIL_RADIUS; k_diff <= CUBE_STENCIL_RADIUS; k_diff++) {
    #pragma unroll
    for (int j_diff = -CUBE_STENCIL_RADIUS; j_diff <= CUBE_STENCIL_RADIUS; j_diff++) {
        #pragma unroll
        for (int i_diff = -CUBE_STENCIL_RADIUS; i_diff <= CUBE_STENCIL_RADIUS; i_diff++) {
            bOut[b][k][j][i] += (bIn[b][k + k_diff][j + j_diff][i + i_diff] * 
                                 coeff[k_diff + CUBE_STENCIL_RADIUS][j_diff + CUBE_STENCIL_RADIUS][i_diff + CUBE_STENCIL_RADIUS]);
        }
    }
  }
  */
}

__global__ void
d3star_brick_codegen(unsigned (*grid)[STRIDEBY][STRIDEBX], Brick <Dim<BDIM>, Dim<VFOLD>> bIn,
                  Brick <Dim<BDIM>, Dim<VFOLD>> bOut){
                  //bElem *coeff) {
  long tk = GB + blockIdx.z;
  long tj = GB + blockIdx.y;
  long ti = GBX + blockIdx.x;
  unsigned b = grid[tk][tj][ti];
  brick("star"+str(STAR_STENCIL_RADIUS)+".py", VSVEC, (BDIM), (VFOLD), b);
}

__global__ void
d3cube_brick_codegen(unsigned (*grid)[STRIDEBY][STRIDEBX], Brick <Dim<BDIM>, Dim<VFOLD>> bIn,
                  Brick <Dim<BDIM>, Dim<VFOLD>> bOut){
                  //bElem (*coeff)[8][8]) {
  long tk = GB + blockIdx.z;
  long tj = GB + blockIdx.y;
  long ti = GBX + blockIdx.x;
  unsigned b = grid[tk][tj][ti];
  brick("cube"+str(CUBE_STENCIL_RADIUS)+".py", VSVEC, (BDIM), (VFOLD), b);
}

__global__ void
d3star_arr(bElem (*arr_in)[STRIDEY][STRIDEX], bElem (*arr_out)[STRIDEY][STRIDEX]){ 
           //bElem *coeff) {
  long k = PADDING + GZ + blockIdx.z * TILE + threadIdx.z;
  long j = PADDING + GZ + blockIdx.y * TILE + threadIdx.y;
  long i = PADDINGX + GZX + blockIdx.x * TILEX + threadIdx.x;
  ST_STAR_ARR_GPU;
  /*
  arr_out[k][j][i] = coeff[0] * arr_in[k][j][i];
  #pragma unroll
  for (int a = 1; a <= STAR_STENCIL_RADIUS; a++) {
      arr_out[k][j][i] += coeff[a] * (
          arr_in[k][j][i + a] + arr_in[k][j + a][i] + arr_in[k + a][j][i] +
          arr_in[k][j][i - a] + arr_in[k][j - a][i] + arr_in[k - a][j][i]);
  }
  */
}

__global__ void
d3cube_arr(bElem (*arr_in)[STRIDEY][STRIDEX], bElem (*arr_out)[STRIDEY][STRIDEX]){ 
           //bElem (*coeff)[8][8]) {
  long k = PADDING + GZ + blockIdx.z * TILE + threadIdx.z;
  long j = PADDING + GZ + blockIdx.y * TILE + threadIdx.y;
  long i = PADDINGX + GZX + blockIdx.x * TILEX + threadIdx.x;
  ST_CUBE_ARR_GPU;
  /*
  arr_out[k][j][i] = 0.0;
  #pragma unroll
  for (int k_diff = -CUBE_STENCIL_RADIUS; k_diff <= CUBE_STENCIL_RADIUS; k_diff++) {
    #pragma unroll
    for (int j_diff = -CUBE_STENCIL_RADIUS; j_diff <= CUBE_STENCIL_RADIUS; j_diff++) {
        #pragma unroll
        for (int i_diff = -CUBE_STENCIL_RADIUS; i_diff <= CUBE_STENCIL_RADIUS; i_diff++) {
            arr_out[k][j][i] += (arr_in[k + k_diff][j + j_diff][i + i_diff] * 
                                 coeff[k_diff + CUBE_STENCIL_RADIUS][j_diff + CUBE_STENCIL_RADIUS][i_diff + CUBE_STENCIL_RADIUS]);
        }
    }
  }
  */
}

#define bIn(i, j, k) arr_in[k][j][i]
#define bOut(i, j, k) arr_out[k][j][i]

__global__ void
d3star_arr_codegen(bElem (*arr_in)[STRIDEY][STRIDEX], bElem (*arr_out)[STRIDEY][STRIDEX]){ 
                   //bElem *coeff) {
  long k = GZ + blockIdx.z * TILE;
  long j = GZ + blockIdx.y * TILE;
  long i = GZX + blockIdx.x * WARPSIZE;
  tile("star"+str(STAR_STENCIL_RADIUS)+".py", VSVEC, (TILE, TILE, WARPSIZE), ("k", "j", "i"), (1, 1, WARPSIZE));
}

__global__ void
d3cube_arr_codegen(bElem (*arr_in)[STRIDEY][STRIDEX], bElem (*arr_out)[STRIDEY][STRIDEX]){ 
                   //bElem (*coeff)[8][8]) {
  long k = GZ + blockIdx.z * TILE;
  long j = GZ + blockIdx.y * TILE;
  long i = GZX + blockIdx.x * WARPSIZE;
  tile("cube"+str(CUBE_STENCIL_RADIUS)+".py", VSVEC, (TILE, TILE, WARPSIZE), ("k", "j", "i"), (1, 1, WARPSIZE));
}

#undef bIn
#undef bOut

void d3_stencils_star_cuda() {
    unsigned *grid_ptr;

    BrickInfo<3> bInfo = init_grid<3>(grid_ptr, {STRIDEBX, STRIDEBY, STRIDEBZ});
    unsigned *grid_dev;
    {
        unsigned size = (STRIDEBX * STRIDEBY * STRIDEBZ) * sizeof(unsigned);
        cudaMalloc(&grid_dev, size);
        cudaMemcpy(grid_dev, grid_ptr, size, cudaMemcpyHostToDevice);
    }
    auto grid = (unsigned (*)[STRIDEBY][STRIDEBX]) grid_dev;

    BrickInfo<3> *binfo_dev = movBrickInfoDeep(bInfo, cudaMemcpyHostToDevice);

    // Create out data
    unsigned data_size = STRIDEX * STRIDEY * STRIDEZ * sizeof(bElem);
    bElem *in_ptr = randomArray({STRIDEX, STRIDEY, STRIDEZ});
    bElem *out_ptr = zeroArray({STRIDEX, STRIDEY, STRIDEZ});
    bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) in_ptr;
    bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) out_ptr;

    // Copy over the coefficient array
    //bElem *coeff_dev;
    //{
    //    unsigned size = COEFF_SIZE * sizeof(bElem);
    //    cudaMalloc(&coeff_dev, size);
    //    cudaMemcpy(coeff_dev, coeff, size, cudaMemcpyHostToDevice);
    //}

    // Copy over the data array
    bElem *in_dev, *out_dev;
    {
        cudaMalloc(&in_dev, data_size);
        cudaMalloc(&out_dev, data_size);
        cudaMemcpy(in_dev, in_ptr, data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(out_dev, out_ptr, data_size, cudaMemcpyHostToDevice);
    }

    // Create the bricks
    auto bsize = cal_size<BDIM>::value;
    auto bstorage = BrickStorage::allocate(bInfo.nbricks, bsize * 2);
    Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bstorage, 0);
    Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bstorage, bsize);

    // Copy data to the bricks
    copyToBrick<3>({STRIDEGX, STRIDEGY, STRIDEGZ}, {PADDINGX, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);
    BrickStorage bstorage_dev = movBrickStorage(bstorage, cudaMemcpyHostToDevice);

    auto arr_func_tile = [&arr_in, &arr_out]() -> void {
        _TILEFOR{ 
          ST_STAR_CPU;
          /*
          arr_out[k][j][i] = coeff[0] * arr_in[k][j][i];
          for (int a = 1; a <= STAR_STENCIL_RADIUS; a++) {
              arr_out[k][j][i] += coeff[a] * (
                  arr_in[k][j][i + a] + arr_in[k][j + a][i] + arr_in[k + a][j][i] +
                  arr_in[k][j][i - a] + arr_in[k][j - a][i] + arr_in[k - a][j][i]);
          }
          */
        }        
    };
    auto brick_func = [&grid, &binfo_dev, &bstorage_dev]() -> void {
        auto bSize = cal_size<BDIM>::value;
        Brick <Dim<BDIM>, Dim<VFOLD>> bIn(binfo_dev, bstorage_dev, 0);
        Brick <Dim<BDIM>, Dim<VFOLD>> bOut(binfo_dev, bstorage_dev, bSize);
        dim3 block(NBX, NBY, NBZ), thread(TILEX,TILE,TILE);
        d3star_brick<<<block,thread>>>(grid, bIn, bOut);
    };
    auto arr_func = [&in_dev, &out_dev]() -> void {
        bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) in_dev;
        bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) out_dev;
        dim3 block(NBX, NBY, NBZ), thread(TILEX,TILE,TILE);
        d3star_arr<<<block,thread>>>(arr_in, arr_out);
    };
    auto arr_func_codegen = [&in_dev, &out_dev]() -> void {
        bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) in_dev;
        bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) out_dev;
        dim3 block(NX / WARPSIZE, NBY, NBZ), thread(WARPSIZE);
        d3star_arr_codegen<<<block,thread>>>(arr_in, arr_out);
    };
    auto brick_func_codegen = [&grid, &binfo_dev, &bstorage_dev]() -> void {
        auto bSize = cal_size<BDIM>::value;
        Brick <Dim<BDIM>, Dim<VFOLD>> bIn(binfo_dev, bstorage_dev, 0);
        Brick <Dim<BDIM>, Dim<VFOLD>> bOut(binfo_dev, bstorage_dev, bSize);
        dim3 block(NBX, NBY, NBZ), thread(WARPSIZE);
        d3star_brick_codegen<<<block,thread>>>(grid, bIn, bOut);
    };

    std::cout << "3D Star stencil with radius " <<STAR_STENCIL_RADIUS<<
                 " aka "<<1+STAR_STENCIL_RADIUS*6<<"pt stencil"<< 
                 " on a "<<NX<<"*"<<NY<<"*"<<NZ<<" domain"<<std::endl;
    arr_func_tile();
    std::cout << "Arr: " << cutime_func(arr_func) << std::endl;
    std::cout << "Arr codegen: " << cutime_func(arr_func_codegen) << std::endl;
    std::cout << "Bri: " << cutime_func(brick_func) << std::endl;
    std::cout << "Bri codegen: " << cutime_func(brick_func_codegen) << std::endl;
    cudaDeviceSynchronize();

    cudaMemcpy(bstorage.dat.get(), bstorage_dev.dat.get(), bstorage.chunks * bstorage.step * sizeof(bElem), cudaMemcpyDeviceToHost);

    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");

    cudaMemcpy(out_ptr, out_dev, bsize, cudaMemcpyDeviceToHost);
    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");

    free(in_ptr);
    free(out_ptr);
    free(grid_ptr);
    free(bInfo.adj);
    cudaFree(in_dev);
    cudaFree(out_dev);
}

void d3_stencils_cube_cuda() {
    unsigned *grid_ptr;

    BrickInfo<3> bInfo = init_grid<3>(grid_ptr, {STRIDEBX, STRIDEBY, STRIDEBZ});
    unsigned *grid_dev;
    {
        unsigned size = (STRIDEBX * STRIDEBY * STRIDEBZ) * sizeof(unsigned);
        cudaMalloc(&grid_dev, size);
        cudaMemcpy(grid_dev, grid_ptr, size, cudaMemcpyHostToDevice);
    }
    auto grid = (unsigned (*)[STRIDEBY][STRIDEBX]) grid_dev;

    BrickInfo<3> *binfo_dev = movBrickInfoDeep(bInfo, cudaMemcpyHostToDevice);

    // Create out data
    unsigned data_size = STRIDEX * STRIDEY * STRIDEZ * sizeof(bElem);
    bElem *in_ptr = randomArray({STRIDEX, STRIDEY, STRIDEZ});
    bElem *out_ptr = zeroArray({STRIDEX, STRIDEY, STRIDEZ});
    bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) in_ptr;
    bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) out_ptr;

    // Copy over the coefficient array
    //bElem *coeff_dev;
    //{
    //    unsigned size = COEFF_SIZE * sizeof(bElem);
    //    cudaMalloc(&coeff_dev, size);
    //    cudaMemcpy(coeff_dev, coeff, size, cudaMemcpyHostToDevice);
    //}

    // Copy over the data array
    bElem *in_dev, *out_dev;
    {
        cudaMalloc(&in_dev, data_size);
        cudaMalloc(&out_dev, data_size);
        cudaMemcpy(in_dev, in_ptr, data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(out_dev, out_ptr, data_size, cudaMemcpyHostToDevice);
    }

    // Create the bricks
    auto bsize = cal_size<BDIM>::value;
    auto bstorage = BrickStorage::allocate(bInfo.nbricks, bsize * 2);
    Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bstorage, 0);
    Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bstorage, bsize);

    // Copy data to the bricks
    copyToBrick<3>({STRIDEGX, STRIDEGY, STRIDEGZ}, {PADDINGX, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);
    BrickStorage bstorage_dev = movBrickStorage(bstorage, cudaMemcpyHostToDevice);

    //auto coeff_cube = (bElem (*)[8][8]) coeff;
    //auto coeff_cube_dev = (bElem (*)[8][8]) coeff_dev;

    auto arr_func_tile = [&arr_in, &arr_out]() -> void {
        _TILEFOR{ 
            ST_CUBE_CPU;
            /*
            arr_out[k][j][i] = 0.0;
            #pragma unroll
            for (int k_diff = -CUBE_STENCIL_RADIUS; k_diff <= CUBE_STENCIL_RADIUS; k_diff++) {
                #pragma unroll
                for (int j_diff = -CUBE_STENCIL_RADIUS; j_diff <= CUBE_STENCIL_RADIUS; j_diff++) {
                    #pragma unroll
                    for (int i_diff = -CUBE_STENCIL_RADIUS; i_diff <= CUBE_STENCIL_RADIUS; i_diff++) {		
                        arr_out[k][j][i] += (arr_in[k + k_diff][j + j_diff][i + i_diff] * 
                                             coeff_cube[k_diff + CUBE_STENCIL_RADIUS][j_diff + CUBE_STENCIL_RADIUS][i_diff + CUBE_STENCIL_RADIUS]);
                    }
                }
            }
            */
        }        
    };
    auto brick_func = [&grid, &binfo_dev, &bstorage_dev]() -> void {
        auto bSize = cal_size<BDIM>::value;
        Brick <Dim<BDIM>, Dim<VFOLD>> bIn(binfo_dev, bstorage_dev, 0);
        Brick <Dim<BDIM>, Dim<VFOLD>> bOut(binfo_dev, bstorage_dev, bSize);
        dim3 block(NBX, NBY, NBZ), thread(TILEX,TILE,TILE);
        d3cube_brick<<<block,thread>>>(grid, bIn, bOut);
    };
    auto arr_func = [&in_dev, &out_dev]() -> void {
        bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) in_dev;
        bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) out_dev;
        dim3 block(NBX, NBY, NBZ), thread(TILEX,TILE,TILE);
        d3cube_arr<<<block,thread>>>(arr_in, arr_out);
    };
    auto arr_func_codegen = [&in_dev, &out_dev]() -> void {
        bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) in_dev;
        bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) out_dev;
        dim3 block(NX / WARPSIZE, NBY, NBZ), thread(WARPSIZE);
        d3cube_arr_codegen<<<block,thread>>>(arr_in, arr_out);
    };
    auto brick_func_codegen = [&grid, &binfo_dev, &bstorage_dev]() -> void {
        auto bSize = cal_size<BDIM>::value;
        Brick <Dim<BDIM>, Dim<VFOLD>> bIn(binfo_dev, bstorage_dev, 0);
        Brick <Dim<BDIM>, Dim<VFOLD>> bOut(binfo_dev, bstorage_dev, bSize);
        dim3 block(NBX, NBY, NBZ), thread(WARPSIZE);
        d3cube_brick_codegen<<<block,thread>>>(grid, bIn, bOut);
    };

    std::cout << "3D Cube stencil with radius " <<CUBE_STENCIL_RADIUS<<
                 " aka "<<pow(1+CUBE_STENCIL_RADIUS*2,3)<<"pt stencil"<< 
                 " on a "<<NX<<"*"<<NY<<"*"<<NZ<<" domain"<<std::endl;
    arr_func_tile();
    std::cout << "Arr: " << cutime_func(arr_func) << std::endl;
    std::cout << "Arr codegen: " << cutime_func(arr_func_codegen) << std::endl;
    std::cout << "Bri: " << cutime_func(brick_func) << std::endl;
    std::cout << "Bri codegen: " << cutime_func(brick_func_codegen) << std::endl;
    cudaDeviceSynchronize();

    cudaMemcpy(bstorage.dat.get(), bstorage_dev.dat.get(), bstorage.chunks * bstorage.step * sizeof(bElem), cudaMemcpyDeviceToHost);

    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");

    cudaMemcpy(out_ptr, out_dev, bsize, cudaMemcpyDeviceToHost);
    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");

    free(in_ptr);
    free(out_ptr);
    free(grid_ptr);
    free(bInfo.adj);
    cudaFree(in_dev);
    cudaFree(out_dev);
}