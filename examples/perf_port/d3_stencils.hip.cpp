#include "stencils/stencils.hip.h"
#include "d3_stencils.hip.h"
#include <iostream>
#include <ctime>
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include "stencils/hipvfold.h"

#include "macros_coeffs.h"

#if defined(CODEGEN_ARCH_NVIDIA)
#undef VSVEC
#define VSVEC "HIP"
#define WARPSIZE 32
#undef VFOLD
#define VFOLD 4, 8
#else
#undef VSVEC
#define VSVEC "HIP-AMD"
#define WARPSIZE 64
#undef VFOLD
#define VFOLD 1, 64
#endif

#undef TILE
#define TILE 4

#undef PADDING
#define PADDING TILE

#undef GZ
#define GZ TILE

#define TILEX 64
#define GZX TILEX
#define PADDINGX TILEX

#undef N 
#define N 512

#define NX N * 1 //16
#define NY N / 1 //4
#define NZ N / 1 //4

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
  long tj = GB + hipBlockIdx_y;
  long tk = GB + hipBlockIdx_z;
  long ti = GBX + hipBlockIdx_x;
  long k = hipThreadIdx_z;
  long j = hipThreadIdx_y;
  long i = hipThreadIdx_x;
  unsigned b = grid[tk][tj][ti];
  ST_STAR_BRICK_GPU;
}

__global__ void
d3cube_brick(unsigned (*grid)[STRIDEBY][STRIDEBX], Brick <Dim<BDIM>, Dim<VFOLD>> bIn, Brick <Dim<BDIM>, Dim<VFOLD>> bOut){
            //bElem (*coeff)[8][8]) {
  long tj = GB + hipBlockIdx_y;
  long tk = GB + hipBlockIdx_z;
  long ti = GBX + hipBlockIdx_x;
  long k = hipThreadIdx_z;
  long j = hipThreadIdx_y;
  long i = hipThreadIdx_x;
  unsigned b = grid[tk][tj][ti];
  ST_CUBE_BRICK_GPU;
}

__global__ void
d3star_brick_codegen(unsigned (*grid)[STRIDEBY][STRIDEBX], Brick <Dim<BDIM>, Dim<VFOLD>> bIn,
                  Brick <Dim<BDIM>, Dim<VFOLD>> bOut){
                  //bElem *coeff) {
  long tk = GB + hipBlockIdx_z;
  long tj = GB + hipBlockIdx_y;
  long ti = GBX + hipBlockIdx_x;
  unsigned b = grid[tk][tj][ti];
  brick("star"+str(STAR_STENCIL_RADIUS)+".py", VSVEC, (BDIM), (VFOLD), b);
}

__global__ void
d3cube_brick_codegen(unsigned (*grid)[STRIDEBY][STRIDEBX], Brick <Dim<BDIM>, Dim<VFOLD>> bIn,
                  Brick <Dim<BDIM>, Dim<VFOLD>> bOut){
                  //bElem (*coeff)[8][8]) {
  long tk = GB + hipBlockIdx_z;
  long tj = GB + hipBlockIdx_y;
  long ti = GBX + hipBlockIdx_x;
  unsigned b = grid[tk][tj][ti];
  brick("cube"+str(CUBE_STENCIL_RADIUS)+".py", VSVEC, (BDIM), (VFOLD), b);
}

__global__ void
d3star_arr(bElem (*arr_in)[STRIDEY][STRIDEX], bElem (*arr_out)[STRIDEY][STRIDEX]){ 
           //bElem *coeff) {
  long k = PADDING + GZ + hipBlockIdx_z * TILE + hipThreadIdx_z;
  long j = PADDING + GZ + hipBlockIdx_y * TILE + hipThreadIdx_y;
  long i = PADDINGX + GZX + hipBlockIdx_x * TILEX + hipThreadIdx_x;
  ST_STAR_ARR_GPU;
}

__global__ void
d3cube_arr(bElem (*arr_in)[STRIDEY][STRIDEX], bElem (*arr_out)[STRIDEY][STRIDEX]){ 
           //bElem (*coeff)[8][8]) {
  long k = PADDING + GZ + hipBlockIdx_z * TILE + hipThreadIdx_z;
  long j = PADDING + GZ + hipBlockIdx_y * TILE + hipThreadIdx_y;
  long i = PADDINGX + GZX + hipBlockIdx_x * TILEX + hipThreadIdx_x;
  ST_CUBE_ARR_GPU;
}

#define bIn(i, j, k) arr_in[k][j][i]
#define bOut(i, j, k) arr_out[k][j][i]

__global__ void
d3star_arr_codegen(bElem (*arr_in)[STRIDEY][STRIDEX], bElem (*arr_out)[STRIDEY][STRIDEX]){ 
                   //bElem *coeff) {
  long k = GZ + hipBlockIdx_z * TILE;
  long j = GZ + hipBlockIdx_y * TILE;
  long i = GZX + hipBlockIdx_x * WARPSIZE;
  tile("star"+str(STAR_STENCIL_RADIUS)+".py", VSVEC, (TILE, TILE, WARPSIZE), ("k", "j", "i"), (1, 1, WARPSIZE));
}

__global__ void
d3cube_arr_codegen(bElem (*arr_in)[STRIDEY][STRIDEX], bElem (*arr_out)[STRIDEY][STRIDEX]){ 
                   //bElem (*coeff)[8][8]) {
  long k = GZ + hipBlockIdx_z * TILE;
  long j = GZ + hipBlockIdx_y * TILE;
  long i = GZX + hipBlockIdx_x * WARPSIZE;
  tile("cube"+str(CUBE_STENCIL_RADIUS)+".py", VSVEC, (TILE, TILE, WARPSIZE), ("k", "j", "i"), (1, 1, WARPSIZE));
}

#undef bIn
#undef bOut

void d3_stencils_star_hip() {
    unsigned *grid_ptr;

    BrickInfo<3> bInfo = init_grid<3>(grid_ptr, {STRIDEBX, STRIDEBY, STRIDEBZ});
    unsigned *grid_dev;
    {
        unsigned size = (STRIDEBX * STRIDEBY * STRIDEBZ) * sizeof(unsigned);
        hipMalloc(&grid_dev, size);
        hipMemcpy(grid_dev, grid_ptr, size, hipMemcpyHostToDevice);
    }
    auto grid = (unsigned (*)[STRIDEBY][STRIDEBX]) grid_dev;

    BrickInfo<3> *binfo_dev = movBrickInfoDeep(bInfo, hipMemcpyHostToDevice);

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
    //    hipMalloc(&coeff_dev, size);
    //    hipMemcpy(coeff_dev, coeff, size, hipMemcpyHostToDevice);
    //}

    // Copy over the data array
    bElem *in_dev, *out_dev;
    {
        hipMalloc(&in_dev, data_size);
        hipMalloc(&out_dev, data_size);
        hipMemcpy(in_dev, in_ptr, data_size, hipMemcpyHostToDevice);
        hipMemcpy(out_dev, out_ptr, data_size, hipMemcpyHostToDevice);
    }

    // Create the bricks
    auto bsize = cal_size<BDIM>::value;
    auto bstorage = BrickStorage::allocate(bInfo.nbricks, bsize * 2);
    Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bstorage, 0);
    Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bstorage, bsize);

    // Copy data to the bricks
    copyToBrick<3>({STRIDEGX, STRIDEGY, STRIDEGZ}, {PADDINGX, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);
    BrickStorage bstorage_dev = movBrickStorage(bstorage, hipMemcpyHostToDevice);

    auto arr_func_tile = [&arr_in, &arr_out]() -> void {
        _TILEFOR{ 
          ST_STAR_CPU;
        }        
    };
    auto brick_func = [&grid, &binfo_dev, &bstorage_dev]() -> void {
        auto bSize = cal_size<BDIM>::value;
        Brick <Dim<BDIM>, Dim<VFOLD>> bIn(binfo_dev, bstorage_dev, 0);
        Brick <Dim<BDIM>, Dim<VFOLD>> bOut(binfo_dev, bstorage_dev, bSize);
        dim3 block(NBX, NBY, NBZ), thread(TILEX,TILE,TILE);
        hipLaunchKernelGGL(d3star_brick, block, thread, 0, 0, 
            grid, bIn, bOut);
    };
    auto arr_func = [&in_dev, &out_dev]() -> void {
        bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) in_dev;
        bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) out_dev;
        dim3 block(NBX, NBY, NBZ), thread(TILEX,TILE,TILE);
        hipLaunchKernelGGL(d3star_arr, block, thread, 0, 0,
            arr_in, arr_out);
    };
    auto arr_func_codegen = [&in_dev, &out_dev]() -> void {
        bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) in_dev;
        bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) out_dev;
        dim3 block(NX / WARPSIZE, NBY, NBZ), thread(WARPSIZE);
        hipLaunchKernelGGL(d3star_arr_codegen, block, thread, 0, 0,
            arr_in, arr_out);
    };
    auto brick_func_codegen = [&grid, &binfo_dev, &bstorage_dev]() -> void {
        auto bSize = cal_size<BDIM>::value;
        Brick <Dim<BDIM>, Dim<VFOLD>> bIn(binfo_dev, bstorage_dev, 0);
        Brick <Dim<BDIM>, Dim<VFOLD>> bOut(binfo_dev, bstorage_dev, bSize);
        dim3 block(NBX, NBY, NBZ), thread(WARPSIZE);
        hipLaunchKernelGGL(d3star_brick_codegen, block, thread, 0, 0,
            grid, bIn, bOut);
    };

    std::cout << "3D Star stencil with radius " <<STAR_STENCIL_RADIUS<<
                 " aka "<<1+STAR_STENCIL_RADIUS*6<<"pt stencil"<< 
                 " on a "<<NX<<"*"<<NY<<"*"<<NZ<<" domain"<<std::endl;
    arr_func_tile();
    std::cout << "Arr: " << hiptime_func(arr_func) << std::endl;
    std::cout << "Arr codegen: " << hiptime_func(arr_func_codegen) << std::endl;
    std::cout << "Bri codegen: " << hiptime_func(brick_func_codegen) << std::endl;
    hipDeviceSynchronize();

    hipMemcpy(bstorage.dat.get(), bstorage_dev.dat.get(), bstorage.chunks * bstorage.step * sizeof(bElem), hipMemcpyDeviceToHost);

    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");

    hipMemcpy(out_ptr, out_dev, bsize, hipMemcpyDeviceToHost);
    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");

    free(in_ptr);
    free(out_ptr);
    free(grid_ptr);
    free(bInfo.adj);
    hipFree(in_dev);
    hipFree(out_dev);
}

void d3_stencils_cube_hip() {
    unsigned *grid_ptr;

    BrickInfo<3> bInfo = init_grid<3>(grid_ptr, {STRIDEBX, STRIDEBY, STRIDEBZ});
    unsigned *grid_dev;
    {
        unsigned size = (STRIDEBX * STRIDEBY * STRIDEBZ) * sizeof(unsigned);
        hipMalloc(&grid_dev, size);
        hipMemcpy(grid_dev, grid_ptr, size, hipMemcpyHostToDevice);
    }
    auto grid = (unsigned (*)[STRIDEBY][STRIDEBX]) grid_dev;

    BrickInfo<3> *binfo_dev = movBrickInfoDeep(bInfo, hipMemcpyHostToDevice);

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
    //    hipMalloc(&coeff_dev, size);
    //    hipMemcpy(coeff_dev, coeff, size, hipMemcpyHostToDevice);
    //}

    // Copy over the data array
    bElem *in_dev, *out_dev;
    {
        hipMalloc(&in_dev, data_size);
        hipMalloc(&out_dev, data_size);
        hipMemcpy(in_dev, in_ptr, data_size, hipMemcpyHostToDevice);
        hipMemcpy(out_dev, out_ptr, data_size, hipMemcpyHostToDevice);
    }

    // Create the bricks
    auto bsize = cal_size<BDIM>::value;
    auto bstorage = BrickStorage::allocate(bInfo.nbricks, bsize * 2);
    Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bstorage, 0);
    Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bstorage, bsize);

    // Copy data to the bricks
    copyToBrick<3>({STRIDEGX, STRIDEGY, STRIDEGZ}, {PADDINGX, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);
    BrickStorage bstorage_dev = movBrickStorage(bstorage, hipMemcpyHostToDevice);

    //auto coeff_cube = (bElem (*)[8][8]) coeff;
    //auto coeff_cube_dev = (bElem (*)[8][8]) coeff_dev;

    auto arr_func_tile = [&arr_in, &arr_out]() -> void {
        _TILEFOR{ 
            ST_CUBE_CPU;
        }        
    };
    auto brick_func = [&grid, &binfo_dev, &bstorage_dev]() -> void {
        auto bSize = cal_size<BDIM>::value;
        Brick <Dim<BDIM>, Dim<VFOLD>> bIn(binfo_dev, bstorage_dev, 0);
        Brick <Dim<BDIM>, Dim<VFOLD>> bOut(binfo_dev, bstorage_dev, bSize);
        dim3 block(NBX, NBY, NBZ), thread(TILEX,TILE,TILE);
        hipLaunchKernelGGL(d3cube_brick, block, thread, 0, 0, 
            grid, bIn, bOut);
    };
    auto arr_func = [&in_dev, &out_dev]() -> void {
        bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) in_dev;
        bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) out_dev;
        dim3 block(NBX, NBY, NBZ), thread(TILEX,TILE,TILE);
        hipLaunchKernelGGL(d3cube_arr, block, thread, 0, 0,
            arr_in, arr_out);
    };
    auto arr_func_codegen = [&in_dev, &out_dev]() -> void {
        bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) in_dev;
        bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem (*)[STRIDEY][STRIDEX]) out_dev;
        dim3 block(NX / WARPSIZE, NBY, NBZ), thread(WARPSIZE);
        hipLaunchKernelGGL(d3cube_arr_codegen, block, thread, 0, 0,
            arr_in, arr_out);
    };
    auto brick_func_codegen = [&grid, &binfo_dev, &bstorage_dev]() -> void {
        auto bSize = cal_size<BDIM>::value;
        Brick <Dim<BDIM>, Dim<VFOLD>> bIn(binfo_dev, bstorage_dev, 0);
        Brick <Dim<BDIM>, Dim<VFOLD>> bOut(binfo_dev, bstorage_dev, bSize);
        dim3 block(NBX, NBY, NBZ), thread(WARPSIZE);
        hipLaunchKernelGGL(d3cube_brick_codegen, block, thread, 0, 0,
            grid, bIn, bOut);
    };

    std::cout << "3D Cube stencil with radius " <<CUBE_STENCIL_RADIUS<<
                 " aka "<<pow(1+CUBE_STENCIL_RADIUS*2,3)<<"pt stencil"<< 
                 " on a "<<NX<<"*"<<NY<<"*"<<NZ<<" domain"<<std::endl;
    arr_func_tile();
    std::cout << "Arr: " << hiptime_func(arr_func) << std::endl;
    std::cout << "Arr codegen: " << hiptime_func(arr_func_codegen) << std::endl;
    std::cout << "Bri codegen: " << hiptime_func(brick_func_codegen) << std::endl;
    hipDeviceSynchronize();

    hipMemcpy(bstorage.dat.get(), bstorage_dev.dat.get(), bstorage.chunks * bstorage.step * sizeof(bElem), hipMemcpyDeviceToHost);

    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");

    hipMemcpy(out_ptr, out_dev, bsize, hipMemcpyDeviceToHost);
    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr, bOut))
        throw std::runtime_error("result mismatch!");

    free(in_ptr);
    free(out_ptr);
    free(grid_ptr);
    free(bInfo.adj);
    hipFree(in_dev);
    hipFree(out_dev);
}
