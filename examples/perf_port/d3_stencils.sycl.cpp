#include "brick-sycl.h"
#include "brick-gpu.h"
#include "brickcompare.h"
#include "multiarray.h"
#include <CL/sycl.hpp>
#include <iostream>

#include "stencils/stencils.h"
#include "macros_coeffs.h"

#if defined (CODEGEN_ARCH_NVIDIA)
#define VSVEC "SYCL-NVIDIA"
#define SYCL_SUBGROUP 32
#undef VFOLD
#define VFOLD 1,32 
#define TILEX 32
#endif
#if defined (CODEGEN_ARCH_AMD)
#define VSVEC "SYCL-AMD"
#define SYCL_SUBGROUP 64
#undef VFOLD
#define VFOLD 1, 64
#define TILEX 64
#endif
#if defined (CODEGEN_ARCH_INTEL)
#define VSVEC "SYCL-INTEL"
#define SYCL_SUBGROUP 16
#undef VFOLD
#define VFOLD 1, 16
#define TILEX 16
#endif

#undef TILE
#define TILE 4

#undef PADDING
#define PADDING TILE

#undef GZ
#define GZ TILE

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

using namespace cl::sycl;

cl::sycl::device *sycl_device;

template <typename T> double sycl_time_func(cl::sycl::queue &squeue, T kernel) {
  int sycl_iter = 100;
  auto st_event = squeue.submit(kernel);
  for (int i = 0; i < sycl_iter - 2; ++i)
    squeue.submit(kernel);
  auto ed_event = squeue.submit(kernel);
  ed_event.wait();
  double elapsed = ed_event.template get_profiling_info<info::event_profiling::command_end>() -
                   st_event.template get_profiling_info<info::event_profiling::command_start>();
  elapsed *= (1e-9 / sycl_iter);
  return elapsed;
}

void printInfo(cl::sycl::device &Device) {
  std::cout << "SYCL test using " << (Device.is_cpu() ? "CPU" : "GPU") << " device {"
            << Device.get_info<cl::sycl::info::device::name>() << "} from {"
            << Device.get_info<cl::sycl::info::device::vendor>() << "}" << std::endl;

  auto dot_num_groups = Device.get_info<cl::sycl::info::device::max_compute_units>();
  auto dot_wgsize = Device.get_info<cl::sycl::info::device::max_work_group_size>();

  std::cout << "Compute units: " << dot_num_groups << std::endl;
  std::cout << "Workgroup size: " << dot_wgsize << std::endl;

  std::cout << "Sub-group Sizes Available on this device: ";
  for (const auto &s :
       Device.get_info<sycl::info::device::sub_group_sizes>()) {
    std::cout << s << " ";
  }
  std::cout << std::endl;
}

void syclinit() {

  class NEOGPUDeviceSelector : public cl::sycl::device_selector  {
  public:
    int operator()(const cl::sycl::device &Device) const override {
      const std::string DeviceName = Device.get_info<cl::sycl::info::device::name>();
      const std::string DeviceVendor = Device.get_info<cl::sycl::info::device::vendor>();

      return (Device.is_gpu() && ((DeviceName.find("NVIDIA") != std::string::npos)
                              || (DeviceName.find("gfx") != std::string::npos)
                              || (DeviceName.find("Intel") != std::string::npos)))
                 ? 1
                 : -1;
    }
  };

  sycl_device = new cl::sycl::device(NEOGPUDeviceSelector());
  printInfo(*sycl_device);
}

void d3_stencils_cube_sycl() {
  unsigned *grid_ptr;

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEBX, STRIDEBY, STRIDEBZ});
  auto grid = (unsigned(*)[STRIDEBY][STRIDEBX])grid_ptr;

  bElem *in_ptr = randomArray({STRIDEX, STRIDEY, STRIDEZ});
  bElem *out_ptr = zeroArray({STRIDEX, STRIDEY, STRIDEZ});
  bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem(*)[STRIDEY][STRIDEX])in_ptr;
  bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem(*)[STRIDEY][STRIDEX])out_ptr;

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  BrickInfo<3> _bInfo_dev = movBrickInfo(bInfo, gpuMemcpyHostToDevice);
  BrickInfo<3> *bInfo_dev;
  {
      unsigned size = sizeof(BrickInfo < 3 > );
      gpuMalloc(&bInfo_dev, size);
      gpuMemcpy(bInfo_dev, &_bInfo_dev, size, gpuMemcpyHostToDevice);
  }
  buffer<BrickInfo<3>, 1> bInfo_buf(&_bInfo_dev, range<1>(sizeof(BrickInfo < 3 > )));

  cl::sycl::queue squeue(*sycl_device, {cl::sycl::property::queue::enable_profiling()});

  copyToBrick<3>({STRIDEGX, STRIDEGY, STRIDEGZ}, {PADDINGX, PADDING, PADDING}, {0, 0, 0}, in_ptr,
                 grid_ptr, bIn);
  // Setup bricks for opencl
  //buffer<bElem, 1> coeff_buf(coeff, range<1>(125));

  std::vector<unsigned> bIdx;

  for (long tk = GB; tk < STRIDEBZ - GB; ++tk)
    for (long tj = GB; tj < STRIDEBY - GB; ++tj)
      for (long ti = GBX; ti < STRIDEBX - GBX; ++ti)
        bIdx.push_back(grid[tk][tj][ti]);

  buffer<unsigned, 1> bIdx_buf(bIdx.data(), range<1>(bIdx.size()));

  size_t adj_size = bInfo.nbricks * 27;
  buffer<unsigned, 1> adj_buf((unsigned *)bInfo.adj, range<1>(adj_size));

  size_t bDat_size = bStorage.chunks * bStorage.step;
  buffer<bElem, 1> bDat_buf({range<1>(bDat_size)});

  //auto coeff_cube = (bElem (*)[5][5]) coeff;
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
        }*/
    }  
  };

  std::cout << "3D Cube stencil with radius " <<CUBE_STENCIL_RADIUS<<
               " aka "<<pow(1+CUBE_STENCIL_RADIUS*2,3)<<"pt stencil"<< 
               " on a "<<NX<<"*"<<NY<<"*"<<NZ<<" domain"<<std::endl;
  arr_func_tile();

  buffer<bElem, 3> arr_in_buf({range<3>(STRIDEZ, STRIDEY, STRIDEX)});
  buffer<bElem, 3> arr_out_buf({range<3>(STRIDEZ, STRIDEY, STRIDEX)});
  //buffer<bElem, 3> coeff_cube_buf({range<3>(5, 5, 5)});

  auto arr_func = [&](handler &cgh) {
    //auto coeff = coeff_cube_buf.get_access<access::mode::read>(cgh);
    auto arr_in = arr_in_buf.get_access<access::mode::read>(cgh);
    auto arr_out = arr_out_buf.get_access<access::mode::write>(cgh);

    nd_range<3> nworkitem(range<3>(NX, NY, NZ), range<3>(TILEX, TILE, TILE));
    cgh.parallel_for<class d3cube_arr>(nworkitem, [=](nd_item<3> WIid) {
      auto i = WIid.get_global_id(0) + GZX + PADDINGX;
      auto j = WIid.get_global_id(1) + GZ + PADDING;
      auto k = WIid.get_global_id(2) + GZ + PADDING;

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
    });
  };

#define bIn(i, j, k) arr_in[k][j][i]
#define bOut(i, j, k) arr_out[k][j][i]

  auto arr_func_codegen = [&](handler &cgh) {
    //auto coeff = coeff_cube_buf.get_access<access::mode::read>(cgh);
    auto arr_in = arr_in_buf.get_access<access::mode::read>(cgh);
    auto arr_out = arr_out_buf.get_access<access::mode::write>(cgh);

    nd_range<3> nworkitem(range<3>(NX, STRIDEBY-2, STRIDEBZ-2), range<3>(SYCL_SUBGROUP, 1, 1));
    cgh.parallel_for<class d3cube_arr_codegen>(
        nworkitem, [=](nd_item<3> WIid) [[intel::reqd_sub_group_size(SYCL_SUBGROUP)]] {
        //nworkitem, [=](nd_item<3> WIid) {

      long k = GZ + PADDING + WIid.get_group(2) * TILE;
      long j = GZ + PADDING + WIid.get_group(1) * TILE;
      long i = GZX + PADDINGX + WIid.get_group(0) * SYCL_SUBGROUP;
      unsigned sglid = WIid.get_local_id(0);
          tile("cube"+str(CUBE_STENCIL_RADIUS)+".py", VSVEC, (TILE, TILE, SYCL_SUBGROUP), ("k", "j", "i"), (1, 1, SYCL_SUBGROUP));
        });
 
  };

#undef bIn
#undef bOut

  {
    // Move to gpu
    squeue.submit([&](cl::sycl::handler &cgh) {
      auto arr_in = arr_in_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.copy(in_ptr, arr_in);
    });
    squeue.wait();
    //squeue.submit([&](cl::sycl::handler &cgh) {
    //  auto coeff_cube = coeff_cube_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    //  cgh.copy(coeff, coeff_cube);
    //});
  }

  {
    // Run sycl arr_func
    std::cout << "Arr sycl: " << sycl_time_func(squeue, arr_func) << std::endl;
    std::cout << "Arr codegen sycl: " << sycl_time_func(squeue, arr_func_codegen) << std::endl;
  }

  auto brick_func = [&](handler &cgh) {
    auto bDat_s = bDat_buf.get_access<access::mode::read_write>(cgh);
    auto adj_s = adj_buf.get_access<access::mode::read_write>(cgh);
    //auto coeff = coeff_cube_buf.get_access<access::mode::read>(cgh);
    auto bIdx_s = bIdx_buf.get_access<access::mode::read>(cgh);
    auto len = bIdx.size();
    auto bInfo_s = bInfo_buf.get_access<access::mode::read>(cgh);

    nd_range<3> nworkitem(range<3>(NX, NY, NZ), range<3>(TILEX, TILE, TILE));
    cgh.parallel_for<class d3cube_brick>(nworkitem, [=](nd_item<3> WIid) {

      long bk = WIid.get_group(2);
      long bj = WIid.get_group(1);
      long bi = WIid.get_group(0);
      long k = WIid.get_local_id(2);
      long j = WIid.get_local_id(1);
      long i = WIid.get_local_id(0);

      bElem *bDat = (bElem *) bDat_s.get_pointer();
      auto bSize = cal_size<BDIM>::value;
      syclBrick<Dim<BDIM>, Dim<VFOLD>> bIn(bInfo_s.get_pointer(), bDat, bSize * 2, 0);
      syclBrick<Dim<BDIM>, Dim<VFOLD>> bOut(bInfo_s.get_pointer(), bDat, bSize * 2, bSize);

      unsigned b = bIdx_s[bi + (bj + bk * (STRIDEBY-2)) * (STRIDEBX-2)];

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
    });
  };

  auto brick_func_codegen = [&](handler &cgh) {
    auto bDat_s = bDat_buf.get_access<access::mode::read_write>(cgh);
    auto adj_s = adj_buf.get_access<access::mode::read_write>(cgh);
    //auto coeff = coeff_buf.get_access<access::mode::read>(cgh);
    auto bIdx_s = bIdx_buf.get_access<access::mode::read>(cgh);
    auto len = bIdx.size();

    nd_range<1> nworkitem(range<1>((STRIDEBX-2) * (STRIDEBY-2) * (STRIDEBZ-2) * SYCL_SUBGROUP), range<1>(SYCL_SUBGROUP));
    cgh.parallel_for<class d3cube_brick_codegen>(
        nworkitem, [=](nd_item<1> WIid) [[intel::reqd_sub_group_size(SYCL_SUBGROUP)]] {
          auto SG = WIid.get_sub_group();
          auto sglid = WIid.get_local_id();

          oclbrick bIn = {bDat_s.get_pointer(), adj_s.get_pointer(), bSize * 2};
          oclbrick bOut = {((bElem *)bDat_s.get_pointer()) + bSize, adj_s.get_pointer(), bSize * 2};
          for (unsigned i = WIid.get_group(0); i < len; i += WIid.get_group_range(0)) {
            unsigned b = bIdx_s[i];
            brick("cube"+str(CUBE_STENCIL_RADIUS)+".py", VSVEC, (TILE, TILE, TILEX), (VFOLD), b);
          }
        });
  };

  {
    // Move to gpu
    squeue.submit([&](cl::sycl::handler &cgh) {
      auto bDat_s = bDat_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.copy(bStorage.dat, bDat_s);
    });
  }

  {
    std::cout << "Brick sycl: " << sycl_time_func(squeue, brick_func) << std::endl;
    std::cout << "Brick codegen sycl: " << sycl_time_func(squeue, brick_func_codegen) << std::endl;
  }

  {
    // Move back from gpu
    squeue.submit([&](cl::sycl::handler &cgh) {
      auto bDat_s = bDat_buf.get_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(bDat_s, bStorage.dat);
    });
  }
  squeue.wait();

  {
    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr,
                         bOut)) {
      std::cout << "result mismatch!" << std::endl;
      // Identify mismatch
      for (long tk = GB; tk < STRIDEBZ - GB; ++tk)
        for (long tj = GB; tj < STRIDEBY - GB; ++tj)
          for (long ti = GBX; ti < STRIDEBX - GBX; ++ti) {
            auto b = grid[tk][tj][ti];
            for (long k = 0; k < TILE; ++k)
              for (long j = 0; j < TILE; ++j)
                for (long i = 0; i < TILEX; ++i) {
                  auto aval = arr_out[tk * TILE + k + PADDING][tj * TILE + j + PADDING]
                                     [ti * TILEX + i + PADDINGX];
                  auto diff = abs(bOut[b][k][j][i] - aval);
                  auto sum = abs(bOut[b][k][j][i]) + abs(aval);
                  if (sum > 1e-6 && diff / sum > 1e-6)
                    std::cout << "mismatch at " << ti * TILEX + i - PADDINGX << " : "
                              << tj * TILE + j - TILE << " : " << tk * TILE + k - TILE << " : "
                              << bOut[b][k][j][i] << " : " << aval <<" brick "<<ti<<" "<<tj<<" "<<tk<< std::endl;
                }
          }
    }
  }

  {
    // Move back from gpu
    squeue.submit([&](cl::sycl::handler &cgh) {
      auto arr_out = arr_out_buf.get_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(arr_out, out_ptr);
    });
  }
  squeue.wait();

  {
    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr,
                         bOut)) {
      std::cout << "result mismatch!" << std::endl;
                    }
  }

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
}

void d3_stencils_star_sycl() {
  unsigned *grid_ptr;

  auto bInfo = init_grid<3>(grid_ptr, {STRIDEBX, STRIDEBY, STRIDEBZ});
  auto grid = (unsigned(*)[STRIDEBY][STRIDEBX])grid_ptr;

  bElem *in_ptr = randomArray({STRIDEX, STRIDEY, STRIDEZ});
  bElem *out_ptr = zeroArray({STRIDEX, STRIDEY, STRIDEZ});
  bElem(*arr_in)[STRIDEY][STRIDEX] = (bElem(*)[STRIDEY][STRIDEX])in_ptr;
  bElem(*arr_out)[STRIDEY][STRIDEX] = (bElem(*)[STRIDEY][STRIDEX])out_ptr;

  auto bSize = cal_size<BDIM>::value;
  auto bStorage = BrickStorage::allocate(bInfo.nbricks, bSize * 2);
  Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bStorage, 0);
  Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bStorage, bSize);

  BrickInfo<3> _bInfo_dev = movBrickInfo(bInfo, gpuMemcpyHostToDevice);
  BrickInfo<3> *bInfo_dev;
  {
      unsigned size = sizeof(BrickInfo < 3 > );
      gpuMalloc(&bInfo_dev, size);
      gpuMemcpy(bInfo_dev, &_bInfo_dev, size, gpuMemcpyHostToDevice);
  }
  buffer<BrickInfo<3>, 1> bInfo_buf(&_bInfo_dev, range<1>(sizeof(BrickInfo < 3 > )));

  cl::sycl::queue squeue(*sycl_device, {cl::sycl::property::queue::enable_profiling()});

  copyToBrick<3>({STRIDEGX, STRIDEGY, STRIDEGZ}, {PADDINGX, PADDING, PADDING}, {0, 0, 0}, in_ptr,
                 grid_ptr, bIn);
  // Setup bricks for opencl
  //buffer<bElem, 1> coeff_buf(coeff, range<1>(129));

  std::vector<unsigned> bIdx;

  for (long tk = GB; tk < STRIDEBZ - GB; ++tk)
    for (long tj = GB; tj < STRIDEBY - GB; ++tj)
      for (long ti = GBX; ti < STRIDEBX - GBX; ++ti)
        bIdx.push_back(grid[tk][tj][ti]);

  buffer<unsigned, 1> bIdx_buf(bIdx.data(), range<1>(bIdx.size()));

  size_t adj_size = bInfo.nbricks * 27;
  buffer<unsigned, 1> adj_buf((unsigned *)bInfo.adj, range<1>(adj_size));

  size_t bDat_size = bStorage.chunks * bStorage.step;
  buffer<bElem, 1> bDat_buf({range<1>(bDat_size)});

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

  std::cout << "3D Star stencil with radius " <<STAR_STENCIL_RADIUS<<
               " aka "<<1+STAR_STENCIL_RADIUS*6<<"pt stencil"<< 
               " on a "<<NX<<"*"<<NY<<"*"<<NZ<<" domain"<<std::endl;
  arr_func_tile();

  buffer<bElem, 3> arr_in_buf({range<3>(STRIDEZ, STRIDEY, STRIDEX)});
  buffer<bElem, 3> arr_out_buf({range<3>(STRIDEZ, STRIDEY, STRIDEX)});

  auto arr_func = [&](handler &cgh) {
    //auto coeff = coeff_buf.get_access<access::mode::read>(cgh);
    auto arr_in = arr_in_buf.get_access<access::mode::read>(cgh);
    auto arr_out = arr_out_buf.get_access<access::mode::write>(cgh);

    nd_range<3> nworkitem(range<3>(NX, NY, NZ), range<3>(TILEX, TILE, TILE));
    cgh.parallel_for<class d3star_arr>(nworkitem, [=](nd_item<3> WIid) {
      auto i = WIid.get_global_id(0) + GZX + PADDINGX;
      auto j = WIid.get_global_id(1) + GZ + PADDING;
      auto k = WIid.get_global_id(2) + GZ + PADDING;
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
    });
  };

#define bIn(i, j, k) arr_in[k][j][i]
#define bOut(i, j, k) arr_out[k][j][i]

  auto arr_func_codegen = [&](handler &cgh) {
    //auto coeff = coeff_buf.get_access<access::mode::read>(cgh);
    auto arr_in = arr_in_buf.get_access<access::mode::read>(cgh);
    auto arr_out = arr_out_buf.get_access<access::mode::write>(cgh);

    nd_range<3> nworkitem(range<3>(NX, STRIDEBY-2, STRIDEBZ-2), range<3>(SYCL_SUBGROUP, 1, 1));
    cgh.parallel_for<class d3star_arr_codegen>(
        nworkitem, [=](nd_item<3> WIid) {

      long k = GZ + PADDING + WIid.get_group(2) * TILE;
      long j = GZ + PADDING + WIid.get_group(1) * TILE;
      long i = GZX + PADDINGX + WIid.get_group(0) * SYCL_SUBGROUP;
      unsigned sglid = WIid.get_local_id(0);
            tile("star"+str(STAR_STENCIL_RADIUS)+".py", VSVEC, (TILE, TILE, SYCL_SUBGROUP), ("k", "j", "i"), (1, 1, SYCL_SUBGROUP));

        });
 
  };

#undef bIn
#undef bOut

  {
    // Move to gpu
    squeue.submit([&](cl::sycl::handler &cgh) {
      auto arr_in = arr_in_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.copy(in_ptr, arr_in);
    });
  }

  {
    // Run sycl arr_func
    std::cout << "Arr sycl: " << sycl_time_func(squeue, arr_func) << std::endl;
    std::cout << "Arr codegen sycl: " << sycl_time_func(squeue, arr_func_codegen) << std::endl;
  }

  auto brick_func = [&](handler &cgh) {
    auto bDat_s = bDat_buf.get_access<access::mode::read_write>(cgh);
    auto adj_s = adj_buf.get_access<access::mode::read_write>(cgh);
    //auto coeff = coeff_buf.get_access<access::mode::read>(cgh);
    auto bIdx_s = bIdx_buf.get_access<access::mode::read>(cgh);
    auto len = bIdx.size();
    auto bInfo_s = bInfo_buf.get_access<access::mode::read>(cgh);

    nd_range<3> nworkitem(range<3>(NX, NY, NZ), range<3>(TILEX, TILE, TILE));
    cgh.parallel_for<class d3star_brick>(nworkitem, [=](nd_item<3> WIid) {

      long bk = WIid.get_group(2);
      long bj = WIid.get_group(1);
      long bi = WIid.get_group(0);
      long k = WIid.get_local_id(2);
      long j = WIid.get_local_id(1);
      long i = WIid.get_local_id(0);

      bElem *bDat = (bElem *) bDat_s.get_pointer();
      auto bSize = cal_size<BDIM>::value;
      syclBrick<Dim<BDIM>, Dim<VFOLD>> bIn(bInfo_s.get_pointer(), bDat, bSize * 2, 0);
      syclBrick<Dim<BDIM>, Dim<VFOLD>> bOut(bInfo_s.get_pointer(), bDat, bSize * 2, bSize);

      unsigned b = bIdx_s[bi + (bj + bk * (STRIDEBY-2)) * (STRIDEBX-2)];
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
    });
  };

  auto brick_func_codegen = [&](handler &cgh) {
    auto bDat_s = bDat_buf.get_access<access::mode::read_write>(cgh);
    auto adj_s = adj_buf.get_access<access::mode::read_write>(cgh);
    //auto coeff = coeff_buf.get_access<access::mode::read>(cgh);
    auto bIdx_s = bIdx_buf.get_access<access::mode::read>(cgh);
    auto len = bIdx.size();

    nd_range<1> nworkitem(range<1>((STRIDEBX-2) * (STRIDEBY-2) * (STRIDEBZ-2) * SYCL_SUBGROUP), range<1>(SYCL_SUBGROUP));
    cgh.parallel_for<class d3star_brick_codegen>(
        nworkitem, [=](nd_item<1> WIid) [[intel::reqd_sub_group_size(SYCL_SUBGROUP)]] {
          auto SG = WIid.get_sub_group();
          auto sglid = WIid.get_local_id();

          oclbrick bIn = {bDat_s.get_pointer(), adj_s.get_pointer(), bSize * 2};
          oclbrick bOut = {((bElem *)bDat_s.get_pointer()) + bSize, adj_s.get_pointer(), bSize * 2};
          for (unsigned i = WIid.get_group(0); i < len; i += WIid.get_group_range(0)) {
            unsigned b = bIdx_s[i];
            brick("star"+str(STAR_STENCIL_RADIUS)+".py", VSVEC, (TILE, TILE, TILEX), (VFOLD), b);
          }
        });
  };

  {
    // Move to gpu
    squeue.submit([&](cl::sycl::handler &cgh) {
      auto bDat_s = bDat_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.copy(bStorage.dat, bDat_s);
    });
  }

  {
    std::cout << "Brick sycl: " << sycl_time_func(squeue, brick_func) << std::endl;
    std::cout << "Brick codegen sycl: " << sycl_time_func(squeue, brick_func_codegen) << std::endl;
  }

  {
    // Move back from gpu
    squeue.submit([&](cl::sycl::handler &cgh) {
      auto bDat_s = bDat_buf.get_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(bDat_s, bStorage.dat);
    });
  }
  squeue.wait();

  {
    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr,
                         bOut)) {
      std::cout << "result mismatch!" << std::endl;
      // Identify mismatch
      for (long tk = GB; tk < STRIDEBZ - GB; ++tk)
        for (long tj = GB; tj < STRIDEBY - GB; ++tj)
          for (long ti = GBX; ti < STRIDEBX - GBX; ++ti) {
            auto b = grid[tk][tj][ti];
            for (long k = 0; k < TILE; ++k)
              for (long j = 0; j < TILE; ++j)
                for (long i = 0; i < TILEX; ++i) {
                  auto aval = arr_out[tk * TILE + k + PADDING][tj * TILE + j + PADDING]
                                     [ti * TILEX + i + PADDINGX];
                  auto diff = abs(bOut[b][k][j][i] - aval);
                  auto sum = abs(bOut[b][k][j][i]) + abs(aval);
                  if (sum > 1e-6 && diff / sum > 1e-6)
                    std::cout << "mismatch at " << ti * TILEX + i - TILEX << " : "
                              << tj * TILE + j - TILE << " : " << tk * TILE + k - TILE << " : "
                              << bOut[b][k][j][i] << " : " << aval <<" brick "<<ti<<" "<<tj<<" "<<tk<< std::endl;
                }
          }
    }
  }

  {
    // Move back from gpu
    squeue.submit([&](cl::sycl::handler &cgh) {
      auto arr_out = arr_out_buf.get_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(arr_out, out_ptr);
    });

  }
  squeue.wait();

  {
    if (!compareBrick<3>({NX, NY, NZ}, {PADDINGX, PADDING, PADDING}, {GZX, GZ, GZ}, out_ptr, grid_ptr,
                         bOut)) {
      std::cout << "result mismatch!" << std::endl;
    }
  }

  free(in_ptr);
  free(out_ptr);
  free(grid_ptr);
}
