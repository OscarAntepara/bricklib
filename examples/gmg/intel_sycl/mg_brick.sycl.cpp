//------------------------------------------------------------------------------------------------------------------------------

#include "mg_brick.h"
#include "mg_common.h"

#undef MPI_ALPHA
#undef MPI_BETA
#define MPI_ALPHA -6
#define MPI_BETA 1

#undef ST_SCRTPT
#define ST_SCRTPT "mpi7pt.py"

#define QUAD_INTERP 0

#define JACOBI_COEFF 1.0/9.0

#define PRE_SMOOTH_ITER 6
#define POST_SMOOTH_ITER 6
#define BOTTOM_SOLVER_ITER 100

using namespace cl::sycl;

cl::sycl::device *sycl_device;

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
}

void exchangeField(BrickStorage& field,
                   BrickStorage& field_dev,
                   std::shared_ptr<BrickDecomp<3, BDIM> >& bDecomp,
                   ExchangeView& ev,
                   double& timer)
{

  double st1 = omp_get_wtime();

//#ifndef GPU_AWARE
//  {
  
    double t_a = omp_get_wtime();
    gpuMemcpy(field.dat.get() + field.step * bDecomp->sep_pos[0],
                field_dev.dat.get() + field.step * bDecomp->sep_pos[0],
                field.step * (bDecomp->sep_pos[1] - bDecomp->sep_pos[0]) * sizeof(bElem),
                gpuMemcpyDeviceToHost);
    double t_b = omp_get_wtime();
    movetime += t_b - t_a;
//#ifdef DECOMP_PAGEUNALIGN
    bDecomp->exchange(field);
//#else
//    ev.exchange();
//#endif
    t_a = omp_get_wtime();
    gpuMemcpy(field_dev.dat.get() + field.step * bDecomp->sep_pos[1],
                field.dat.get() + field.step * bDecomp->sep_pos[1],
                field.step * (bDecomp->sep_pos[2] - bDecomp->sep_pos[1]) * sizeof(bElem),
                gpuMemcpyHostToDevice);
    t_b = omp_get_wtime();
    movetime += t_b - t_a;
//  }
//#else*/
  //bDecomp->exchange(field_dev);
//#endif
  timer+= omp_get_wtime() - st1;

}

void 
reduction_kernel(sycl::nd_item<3> item, unsigned *grid, Brick_SYCL_3D res, unsigned *stride, bElem *d_res, bElem *cache)
{

  // accumulate per thread first (multiple elements)
  bElem thread_val = 0.0;
  long bk = 1 + item.get_group(2);
  long bj = 1 + item.get_group(1);
  long bi = 1 + item.get_group(0);
  long k = item.get_local_id(2);
  long j = item.get_local_id(1);
  long i = item.get_local_id(0);

  unsigned b = grid[bi + (bj + bk * stride[1]) * stride[0]];

  if (bi < (stride[1] - 1) && bj < (stride[1] - 1) && bk < (stride[1] - 1)) {
    double val = res[b][k][j][i];
    thread_val = max(thread_val, fabs(val));
  }

  int cacheIndex = item.get_local_id(0) + item.get_local_id(1)*TILE + item.get_local_id(2)*TILE*TILE;
  cache[cacheIndex] = thread_val;   // set the cache value 

  item.barrier(sycl::access::fence_space::local_space);

  // perform parallel reduction, threadsPerBlock must be 2^m

  int ib = (item.get_local_range().get(0)*item.get_local_range().get(1)*item.get_local_range().get(2)) / 2;
  while (ib != 0) {
    if(cacheIndex < ib && cache[cacheIndex + ib] > cache[cacheIndex])
      cache[cacheIndex] = cache[cacheIndex + ib]; 

    item.barrier(sycl::access::fence_space::local_space);

    ib /= 2;
  }

  if (cacheIndex == 0) {
      auto v = sycl::atomic_ref<bElem,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>(
          d_res[0]);
      v.fetch_max(cache[0]);
  }
  
}

void 
applyOp_kernel(sycl::nd_item<1> item, unsigned *grid, oclbrick x, oclbrick Ax, unsigned *stride) {
  auto SG = item.get_sub_group();
  auto sglid = item.get_local_id();
  unsigned len = stride[0]*stride[1]*stride[2];
  for (unsigned i = item.get_group(0); i < len; i += item.get_group_range(0)) {
    unsigned b = grid[i];
    brick(ST_SCRTPT, VSVEC, (BDIM), (VFOLD), b);
  }
}

void 
smooth_residual_kernel(sycl::nd_item<3> item, unsigned *grid, Brick_SYCL_3D x, Brick_SYCL_3D Ax, 
                       Brick_SYCL_3D rhs, Brick_SYCL_3D res, unsigned *stride, bElem* dom_len_dev) {
  long bk = item.get_group(2);
  long bj = item.get_group(1);
  long bi = item.get_group(0);
  long k = item.get_local_id(2);
  long j = item.get_local_id(1);
  long i = item.get_local_id(0);
  unsigned b = grid[bi + (bj + bk * stride[1]) * stride[0]];

  x[b][k][j][i] += JACOBI_COEFF * (Ax[b][k][j][i] - (dom_len_dev[0] * rhs[b][k][j][i]));
  res[b][k][j][i] = rhs[b][k][j][i] - (Ax[b][k][j][i]*dom_len_dev[1]);
}

void 
smooth_kernel(sycl::nd_item<3> item, unsigned *grid, Brick_SYCL_3D x, Brick_SYCL_3D Ax, 
                       Brick_SYCL_3D rhs, unsigned *stride, bElem* dom_len_dev) {
  long bk = item.get_group(2);
  long bj = item.get_group(1);
  long bi = item.get_group(0);
  long k = item.get_local_id(2);
  long j = item.get_local_id(1);
  long i = item.get_local_id(0);
  unsigned b = grid[bi + (bj + bk * stride[1]) * stride[0]];

  x[b][k][j][i] += JACOBI_COEFF * (Ax[b][k][j][i] - (dom_len_dev[0] * rhs[b][k][j][i]));
}

void
restriction_kernel(sycl::nd_item<3> item, unsigned *grid, unsigned *gridCoar,
                 Brick_SYCL_3D res, Brick_SYCL_3D rhs, unsigned *stride, unsigned *strideCoar) {

  long bk = 1 + item.get_group(2);
  long bj = 1 + item.get_group(1);
  long bi = 1 + item.get_group(0);
  long bFk = 1 + item.get_group(2)*2;
  long bFj = 1 + item.get_group(1)*2;
  long bFi = 1 + item.get_group(0)*2;
  long k = item.get_local_id(2);
  long j = item.get_local_id(1);
  long i = item.get_local_id(0);

  unsigned ib = grid[bFi + (bFj + bFk * stride[1]) * stride[0]];
  unsigned ob = gridCoar[bi + (bj + bk * strideCoar[1]) * strideCoar[0]];
  if (bi < (strideCoar[1] - 1) && bj < (strideCoar[1] - 1) && bk < (strideCoar[1] - 1)){
    rhs[ob][k][j][i] =
      (res[ib][k * 2][j * 2][i * 2] + res[ib][k * 2][j * 2][i * 2 + 1] +
       res[ib][k * 2][j * 2 + 1][i * 2] + res[ib][k * 2][j * 2 + 1][i * 2 + 1] +
       res[ib][k * 2 + 1][j * 2][i * 2] + res[ib][k * 2 + 1][j * 2][i * 2 + 1] +
       res[ib][k * 2 + 1][j * 2 + 1][i * 2] + res[ib][k * 2 + 1][j * 2 + 1][i * 2 + 1]) / 8;
  }      
}

void
interpolation_incr_kernel(sycl::nd_item<3> item, unsigned *grid, unsigned *gridCoar,
                         Brick_SYCL_3D xCoar, Brick_SYCL_3D x, unsigned *stride, unsigned *strideCoar) {

  long bk = 1 + item.get_group(2);
  long bj = 1 + item.get_group(1);
  long bi = 1 + item.get_group(0);
  long bFk = 1 + item.get_group(2)*2;
  long bFj = 1 + item.get_group(1)*2;
  long bFi = 1 + item.get_group(0)*2;
  long k = item.get_local_id(2);
  long j = item.get_local_id(1);
  long i = item.get_local_id(0);

  unsigned ob = grid[bFi + (bFj + bFk * stride[1]) * stride[0]];
  unsigned ib = gridCoar[bi + (bj + bk * strideCoar[1]) * strideCoar[0]];
  if (bi < (strideCoar[1] - 1) && bj < (strideCoar[1] - 1) && bk < (strideCoar[1] - 1)){
    x[ob][k * 2][j * 2][i * 2]             +=  xCoar[ib][k][j][i];
    x[ob][k * 2][j * 2][i * 2 + 1]         +=  xCoar[ib][k][j][i];
    x[ob][k * 2][j * 2 + 1][i * 2]         +=  xCoar[ib][k][j][i];
    x[ob][k * 2][j * 2 + 1][i * 2 + 1]     +=  xCoar[ib][k][j][i];
    x[ob][k * 2 + 1][j * 2][i * 2]         +=  xCoar[ib][k][j][i];
    x[ob][k * 2 + 1][j * 2][i * 2 + 1]     +=  xCoar[ib][k][j][i];
    x[ob][k * 2 + 1][j * 2 + 1][i * 2]     +=  xCoar[ib][k][j][i];
    x[ob][k * 2 + 1][j * 2 + 1][i * 2 + 1] +=  xCoar[ib][k][j][i];

  }
}

void init_mg_brick(MG_brick& mg, int num_levels, MPI_Comm cart,int* coo, std::vector<unsigned>& dom_size, int amr_levels){
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  mg.mpi_size = size;

  mg.num_levels = num_levels;
  mg.levels = (levels_brick**)malloc(num_levels*sizeof(levels_brick*));
  syclinit();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank==0) printInfo(*sycl_device);

  for (int ilevel = 0; ilevel < num_levels; ilevel++)
  {
    mg.levels[ilevel] = (levels_brick*)malloc(sizeof(levels_brick));
    long pow_two = std::pow(2,ilevel);
    std::vector<long> stride(3), strideb(3), strideg(3);
    std::vector<unsigned> dom_size_bDecomp;
    dom_size_bDecomp.resize(3);
    for (int i = 0; i < 3; ++i) {
      dom_size_bDecomp[i] = dom_size[i]/pow_two;
      stride[i] = (dom_size[i]/pow_two) + 2 * TILE + 2 * GZ;
      strideg[i] = (dom_size[i]/pow_two) + 2 * TILE;
      strideb[i] = strideg[i] / TILE;      
    }
    mg.multilevel_stride.push_back(stride);
    mg.multilevel_strideg.push_back(strideg);
    mg.multilevel_strideb.push_back(strideb);

    mg.levels[ilevel]->timers.pr = 0.0;
    mg.levels[ilevel]->timers.apply_op = 0.0;
    mg.levels[ilevel]->timers.restriction = 0.0;
    mg.levels[ilevel]->timers.interpolation_incr = 0.0;
    mg.levels[ilevel]->timers.exchange_total = 0.0;
    mg.levels[ilevel]->timers.maxNormRes = 0.0;

    mg.levels[ilevel]->x_ptr = nullptr;
    mg.levels[ilevel]->Ax_ptr = nullptr;
    mg.levels[ilevel]->rhs_ptr = nullptr;
    mg.levels[ilevel]->res_ptr = nullptr;
    mg.levels[ilevel]->x_ptr = zeroArray({stride[0], stride[1], stride[2]});      
    mg.levels[ilevel]->Ax_ptr = zeroArray({stride[0], stride[1], stride[2]});      
    mg.levels[ilevel]->res_ptr = zeroArray({stride[0], stride[1], stride[2]});      
    bElem rhsCoef = 1.0;
    mg.levels[ilevel]->rhs_ptr = prodSinArray(rhsCoef, stride, PADDING, GZ);

    std::shared_ptr<BrickDecomp<3, BDIM>> bDecomp(new BrickDecomp<3, BDIM>(dom_size_bDecomp, GZ));
    bDecomp->comm = cart;
    populate(cart, *bDecomp, 0, 1, coo);
    bDecomp->initialize(skin3d_good);
    mg.multilevel_bDecomp.push_back(bDecomp);

    auto bSize = cal_size<BDIM>::value;
    mg.levels[ilevel]->bInfo = new BrickInfo<3>(bDecomp->getBrickInfo());
#ifdef DECOMP_PAGEUNALIGN
    mg.levels[ilevel]->bStorageX = new BrickStorage(mg.levels[ilevel]->bInfo->allocate(bSize));
    mg.levels[ilevel]->bStorageRhs = new BrickStorage(mg.levels[ilevel]->bInfo->allocate(bSize));
    mg.levels[ilevel]->bStorageAx = new BrickStorage(mg.levels[ilevel]->bInfo->allocate(bSize));
    mg.levels[ilevel]->bStorageRes = new BrickStorage(mg.levels[ilevel]->bInfo->allocate(bSize));
#else
    mg.levels[ilevel]->bStorageX = new BrickStorage(mg.levels[ilevel]->bInfo->mmap_alloc(bSize));
    mg.levels[ilevel]->bStorageRhs = new BrickStorage(mg.levels[ilevel]->bInfo->mmap_alloc(bSize));
    mg.levels[ilevel]->bStorageAx = new BrickStorage(mg.levels[ilevel]->bInfo->mmap_alloc(bSize));
    mg.levels[ilevel]->bStorageRes = new BrickStorage(mg.levels[ilevel]->bInfo->mmap_alloc(bSize));
#endif

    mg.levels[ilevel]->grid_ptr = nullptr;

    mg.levels[ilevel]->grid_ptr = (unsigned *)malloc(sizeof(unsigned) * strideb[2] * strideb[1] * strideb[0]);
    auto grid = (unsigned(*)[strideb[1]][strideb[0]])mg.levels[ilevel]->grid_ptr;

    for (long k = 0; k < strideb[2]; ++k)
      for (long j = 0; j < strideb[1]; ++j)
        for (long i = 0; i < strideb[0]; ++i) {
          grid[k][j][i] = (*bDecomp)[k][j][i];
        }

    for (long k = 1; k < strideb[2] - 1; ++k)
      for (long j = 1; j < strideb[1] - 1; ++j)
        for (long i = 1; i < strideb[0] - 1; ++i) {
          auto l = grid[k][j][i];
          for (long id = 0; id < 27; ++id)
            if (mg.levels[ilevel]->bInfo->adj[mg.levels[ilevel]->bInfo->adj[l][id]][26 - id] != l)
              throw std::runtime_error("err");
        }

    mg.levels[ilevel]->bX = new Brick3D (&(*mg.levels[ilevel]->bInfo), (*mg.levels[ilevel]->bStorageX), 0);
    mg.levels[ilevel]->bAx = new Brick3D (&(*mg.levels[ilevel]->bInfo), (*mg.levels[ilevel]->bStorageAx), 0);
    mg.levels[ilevel]->bRhs = new Brick3D (&(*mg.levels[ilevel]->bInfo), (*mg.levels[ilevel]->bStorageRhs), 0);
    mg.levels[ilevel]->bRes = new Brick3D (&(*mg.levels[ilevel]->bInfo), (*mg.levels[ilevel]->bStorageRes), 0);


    copyToBrick<3>(strideg, {PADDING, PADDING, PADDING}, {0, 0, 0}, mg.levels[ilevel]->x_ptr, 
                   mg.levels[ilevel]->grid_ptr, (*mg.levels[ilevel]->bX));
    copyToBrick<3>(strideg, {PADDING, PADDING, PADDING}, {0, 0, 0}, mg.levels[ilevel]->rhs_ptr, 
                   mg.levels[ilevel]->grid_ptr, (*mg.levels[ilevel]->bRhs));
    copyToBrick<3>(strideg, {PADDING, PADDING, PADDING}, {0, 0, 0}, mg.levels[ilevel]->res_ptr, 
                   mg.levels[ilevel]->grid_ptr, (*mg.levels[ilevel]->bRes));

    // setup brick on device
    auto _bInfo_dev = movBrickInfo((*mg.levels[ilevel]->bInfo), gpuMemcpyHostToDevice);
    {
      unsigned size = sizeof(BrickInfo<3>);
      gpuMalloc(&mg.levels[ilevel]->bInfo_dev, size);
      gpuMemcpy(mg.levels[ilevel]->bInfo_dev, &_bInfo_dev, size, gpuMemcpyHostToDevice);
    }

    mg.levels[ilevel]->bStorageX_dev = new BrickStorage (movBrickStorage((*mg.levels[ilevel]->bStorageX), gpuMemcpyHostToDevice));
    mg.levels[ilevel]->bStorageAx_dev = new BrickStorage (movBrickStorage((*mg.levels[ilevel]->bStorageAx), gpuMemcpyHostToDevice));
    mg.levels[ilevel]->bStorageRhs_dev = new BrickStorage (movBrickStorage((*mg.levels[ilevel]->bStorageRhs), gpuMemcpyHostToDevice));
    mg.levels[ilevel]->bStorageRes_dev = new BrickStorage (movBrickStorage((*mg.levels[ilevel]->bStorageRes), gpuMemcpyHostToDevice));

    mg.levels[ilevel]->bX_dev = new Brick3D (mg.levels[ilevel]->bInfo_dev, (*mg.levels[ilevel]->bStorageX_dev), 0);
    mg.levels[ilevel]->bAx_dev = new Brick3D (mg.levels[ilevel]->bInfo_dev, (*mg.levels[ilevel]->bStorageAx_dev), 0);
    mg.levels[ilevel]->bRhs_dev = new Brick3D (mg.levels[ilevel]->bInfo_dev, (*mg.levels[ilevel]->bStorageRhs_dev), 0);
    mg.levels[ilevel]->bRes_dev = new Brick3D (mg.levels[ilevel]->bInfo_dev, (*mg.levels[ilevel]->bStorageRes_dev), 0);

    mg.levels[ilevel]->grid_dev_ptr = nullptr;
    mg.levels[ilevel]->grid_stride_dev = nullptr;
    copyToDevice(strideb, mg.levels[ilevel]->grid_dev_ptr, mg.levels[ilevel]->grid_ptr);

    {
      unsigned grid_stride_tmp[3];
      for (int i = 0; i < 3; ++i)
        grid_stride_tmp[i] = strideb[i];
      copyToDevice({3}, mg.levels[ilevel]->grid_stride_dev, grid_stride_tmp);
    }

    unsigned ngBricks = 0;
    for (int i = 0; i < bDecomp->ghost.size(); ++i) {
      ngBricks += (bDecomp->skin[i].len - bDecomp->skin[i].first_pad - bDecomp->skin[i].last_pad);
    }

    unsigned ngBricksBytes = ngBricks*2*sizeof(unsigned);
    mg.levels[ilevel]->dd_ptr = (unsigned *)malloc(ngBricksBytes);
    gpuMalloc((void **) &mg.levels[ilevel]->dd_dev_ptr, ngBricksBytes);
    int ngBricks_acum = 0;
    for (int i = 0; i < bDecomp->ghost.size(); ++i) {
      int nBricks = (bDecomp->skin[i].len - bDecomp->skin[i].first_pad - bDecomp->skin[i].last_pad);
      for (int ib = 0; ib < nBricks; ++ib){ 
        int iter = ib*2;
        mg.levels[ilevel]->dd_ptr[ngBricks_acum+iter] = (bDecomp->ghost[i].pos + bDecomp->ghost[i].first_pad)+(ib);
        mg.levels[ilevel]->dd_ptr[ngBricks_acum+(iter+1)] = (bDecomp->skin[i].pos + bDecomp->skin[i].first_pad)+(ib);
      }
      ngBricks_acum+=nBricks*2;
    }
    gpuMemcpy(mg.levels[ilevel]->dd_dev_ptr, mg.levels[ilevel]->dd_ptr, ngBricksBytes, gpuMemcpyHostToDevice);

    bElem dom_len_ptr[2];
    dom_len_ptr[0] = 1.0;
    mg.levels[ilevel]->dom_len_dev = nullptr;
    bElem dom_size = TILE * (mg.multilevel_strideb[ilevel][0] - 2);
    bElem h2 = dom_len_ptr[0]*dom_len_ptr[0]/(dom_size*dom_size);

    dom_len_ptr[0] = h2;
    dom_len_ptr[1] = 1.0/h2;
    gpuMalloc((void**)&mg.levels[ilevel]->dom_len_dev,2*sizeof(bElem));
    gpuMemcpy(mg.levels[ilevel]->dom_len_dev,dom_len_ptr,2*sizeof(bElem),gpuMemcpyHostToDevice);
    

#ifndef DECOMP_PAGEUNALIGN
    mg.levels[ilevel]->ev = new ExchangeView (bDecomp->exchangeView((*mg.levels[ilevel]->bStorageX)));
#endif

  }
}

void point_relax(MG_brick& mgb, int level, dim3 block, dim3 thread, dim3 thread_a, int n_iter, int max_iter){
  cl::sycl::queue squeue(*sycl_device, {cl::sycl::property::queue::enable_profiling()});
  double st1 = omp_get_wtime();

  squeue.submit([&](sycl::handler& cgh) {
      nd_range<1> nworkitem(range<1>(block[0] * block[1] * block[2] * SYCL_SUBGROUP), range<1>(SYCL_SUBGROUP));
      cgh.parallel_for<class applyOp>(nworkitem, 
          [=, grid = mgb.levels[level]->grid_dev_ptr,
           bInfo = mgb.levels[level]->bInfo_dev,
           bXDat = mgb.levels[level]->bStorageX_dev->dat.get(),
           bAxDat = mgb.levels[level]->bStorageAx_dev->dat.get(),
           stride = mgb.levels[level]->grid_stride_dev]
          (sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SYCL_SUBGROUP)]] {
          auto bSize = cal_size<BDIM>::value;
          oclbrick bX = {bXDat, (unsigned *)bInfo->adj, bSize};
          oclbrick bAx = {bAxDat, (unsigned *)bInfo->adj, bSize};
          applyOp_kernel(item, grid, bX, bAx, stride);
      });
  });
  squeue.wait();

  mgb.levels[level]->timers.apply_op += omp_get_wtime() - st1;
  mgb.levels[level]->num_operations.apply_op += 1;
  double st2 = omp_get_wtime();
  if (n_iter==max_iter-1){ 
  squeue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class smooth_residual>(sycl::nd_range<3>(block * thread_a, thread_a), 
          [=, grid = mgb.levels[level]->grid_dev_ptr,
            bInfo = mgb.levels[level]->bInfo_dev,
            bXDat = mgb.levels[level]->bStorageX_dev->dat.get(),
            bAxDat = mgb.levels[level]->bStorageAx_dev->dat.get(),
            bRhsDat = mgb.levels[level]->bStorageRhs_dev->dat.get(),
            bResDat = mgb.levels[level]->bStorageRes_dev->dat.get(),
            stride = mgb.levels[level]->grid_stride_dev,
            dom_lev = mgb.levels[level]->dom_len_dev]
          (sycl::nd_item<3> item){
          auto bSize = cal_size<BDIM>::value;
          Brick_SYCL_3D bX(bInfo, bXDat, bSize, 0);
          Brick_SYCL_3D bAx(bInfo, bAxDat, bSize, 0);
          Brick_SYCL_3D bRhs(bInfo, bRhsDat, bSize, 0);
          Brick_SYCL_3D bRes(bInfo, bResDat, bSize, 0);
          smooth_residual_kernel(item, grid, bX, bAx, bRhs, bRes, stride, dom_lev);
      });
  });
  }else{
  squeue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class smooth>(sycl::nd_range<3>(block * thread_a, thread_a), 
          [=, grid = mgb.levels[level]->grid_dev_ptr,
            bInfo = mgb.levels[level]->bInfo_dev,
            bXDat = mgb.levels[level]->bStorageX_dev->dat.get(),
            bAxDat = mgb.levels[level]->bStorageAx_dev->dat.get(),
            bRhsDat = mgb.levels[level]->bStorageRhs_dev->dat.get(),
            stride = mgb.levels[level]->grid_stride_dev,
            dom_lev = mgb.levels[level]->dom_len_dev]
          (sycl::nd_item<3> item) {
          auto bSize = cal_size<BDIM>::value;
          Brick_SYCL_3D bX(bInfo, bXDat, bSize, 0);
          Brick_SYCL_3D bAx(bInfo, bAxDat, bSize, 0);
          Brick_SYCL_3D bRhs(bInfo, bRhsDat, bSize, 0);
          smooth_kernel(item, grid, bX, bAx, bRhs, stride, dom_lev);
      });
  });    
  }
  squeue.wait();

  mgb.levels[level]->timers.pr += omp_get_wtime() - st2;
  mgb.levels[level]->num_operations.pr += 1;
};

void restriction(MG_brick& mgb, int level, dim3 block, dim3 thread){
  cl::sycl::queue squeue(*sycl_device, {cl::sycl::property::queue::enable_profiling()});
  double st1 = omp_get_wtime();

  squeue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class restriction>(sycl::nd_range<3>(block * thread, thread), 
          [=, gridF = mgb.levels[level]->grid_dev_ptr,
            gridC = mgb.levels[level+1]->grid_dev_ptr,
            bInfoF = mgb.levels[level]->bInfo_dev,
            bInfoC = mgb.levels[level+1]->bInfo_dev,
            bRhsDat = mgb.levels[level+1]->bStorageRhs_dev->dat.get(),
            bResDat = mgb.levels[level]->bStorageRes_dev->dat.get(),
            strideF = mgb.levels[level]->grid_stride_dev,
            strideC = mgb.levels[level+1]->grid_stride_dev]
          (sycl::nd_item<3> item) {
          auto bSize = cal_size<BDIM>::value;
          Brick_SYCL_3D bRhs(bInfoC, bRhsDat, bSize, 0);
          Brick_SYCL_3D bRes(bInfoF, bResDat, bSize, 0);
          restriction_kernel(item, gridF, gridC, bRes, bRhs, strideF, strideC);
      });
  });
  squeue.wait();

  mgb.levels[level]->timers.restriction += omp_get_wtime() - st1;
  mgb.levels[level]->num_operations.restriction += 1;
};

void interpolation_incr(MG_brick& mgb, int level, dim3 block, dim3 thread, int interp_type){
  cl::sycl::queue squeue(*sycl_device, {cl::sycl::property::queue::enable_profiling()});
  double st1 = omp_get_wtime();
  squeue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class interpolation_incr>(sycl::nd_range<3>(block * thread, thread), 
          [=, gridC = mgb.levels[level]->grid_dev_ptr,
            gridF = mgb.levels[level-1]->grid_dev_ptr,
            bXfDat = mgb.levels[level-1]->bStorageX_dev->dat.get(),
            bXcDat = mgb.levels[level]->bStorageX_dev->dat.get(),
            bInfoC = mgb.levels[level]->bInfo_dev,
            bInfoF = mgb.levels[level-1]->bInfo_dev,
            strideC = mgb.levels[level]->grid_stride_dev,
            strideF = mgb.levels[level-1]->grid_stride_dev]
          (sycl::nd_item<3> item) {
          auto bSize = cal_size<BDIM>::value;
          Brick_SYCL_3D bXf(bInfoF, bXfDat, bSize, 0);
          Brick_SYCL_3D bXc(bInfoC, bXcDat, bSize, 0);
          interpolation_incr_kernel(item, gridF, gridC, bXc, bXf, strideF, strideC);
      });
  });
  squeue.wait();

  mgb.levels[level]->timers.interpolation_incr += omp_get_wtime() - st1;
  mgb.levels[level]->num_operations.interpolation_incr += 1;
};

void exchange_pointRelax(MG_brick& mgb, int level, dim3 block, dim3 thread, dim3 thread_a, int n_iter){
  cl::sycl::queue squeue(*sycl_device, {cl::sycl::property::queue::enable_profiling()});
      long pow_two = std::pow(2,level);
      //squeue.wait();
      for (int i = 0; i < n_iter; ++i) {
if (i%4==0){
        double st1 = omp_get_wtime();

//#ifndef GPU_AWARE
        {
          double t_a = omp_get_wtime();
          gpuMemcpy((*mgb.levels[level]->bStorageX).dat.get() + (*mgb.levels[level]->bStorageX).step * mgb.multilevel_bDecomp[level]->sep_pos[0],
                     (*mgb.levels[level]->bStorageX_dev).dat.get() + (*mgb.levels[level]->bStorageX).step * mgb.multilevel_bDecomp[level]->sep_pos[0],
                     (*mgb.levels[level]->bStorageX).step * (mgb.multilevel_bDecomp[level]->sep_pos[1] - mgb.multilevel_bDecomp[level]->sep_pos[0]) * sizeof(bElem),
                     gpuMemcpyDeviceToHost);
          double t_b = omp_get_wtime();
          movetime += t_b - t_a;
//#ifdef DECOMP_PAGEUNALIGN
          mgb.multilevel_bDecomp[level]->exchange((*mgb.levels[level]->bStorageX));
//#else
//          (*mgb.levels[level]->ev).exchange();
//#endif
          t_a = omp_get_wtime();
          gpuMemcpy((*mgb.levels[level]->bStorageX_dev).dat.get() + (*mgb.levels[level]->bStorageX).step * mgb.multilevel_bDecomp[level]->sep_pos[1],
                     (*mgb.levels[level]->bStorageX).dat.get() + (*mgb.levels[level]->bStorageX).step * mgb.multilevel_bDecomp[level]->sep_pos[1],
                     (*mgb.levels[level]->bStorageX).step * (mgb.multilevel_bDecomp[level]->sep_pos[2] - mgb.multilevel_bDecomp[level]->sep_pos[1]) * sizeof(bElem),
                     gpuMemcpyHostToDevice);
          t_b = omp_get_wtime();
          movetime += t_b - t_a;
        }
//#else*/
        //mgb.multilevel_bDecomp[level]->exchange((*mgb.levels[level]->bStorageX_dev));
//#endif
        mgb.levels[level]->timers.exchange_total+= omp_get_wtime() - st1;
        mgb.levels[level]->num_operations.exchange_total+=1;
}
        //why do we need 2 syncs? This works but further investigation is needed.
      //squeue.wait();
        point_relax(mgb,level,block,thread,thread_a,i,n_iter);
      //squeue.wait();

      }
};

void 
initX_kernel(sycl::nd_item<3> item, unsigned *grid, Brick_SYCL_3D x, unsigned *stride) {
  long bk = item.get_group(2);
  long bj = item.get_group(1);
  long bi = item.get_group(0);
  long k = item.get_local_id(2);
  long j = item.get_local_id(1);
  long i = item.get_local_id(0);

  unsigned b = grid[bi + (bj + bk * stride[1]) * stride[0]];

  x[b][k][j][i] = 0.0;
}

void vcycle_brick(MG_brick& mgb, int start_lvl){
  cl::sycl::queue squeue(*sycl_device, {cl::sycl::property::queue::enable_profiling()});

  for (int ilevel = start_lvl; ilevel < mgb.num_levels-1; ilevel++)
  {
    int PR_iter_r = PRE_SMOOTH_ITER;
    dim3 thread_a(BDIM);
    dim3 block(mgb.multilevel_strideb[ilevel][0], mgb.multilevel_strideb[ilevel][1], mgb.multilevel_strideb[ilevel][2]), 
         thread(SYCL_SUBGROUP,1,1);
    //Point-relaxation
    exchange_pointRelax(mgb,ilevel,block,thread,thread_a,PR_iter_r);
    //Restriction
    dim3 blockCoar(mgb.multilevel_strideb[ilevel+1][0], mgb.multilevel_strideb[ilevel+1][1], mgb.multilevel_strideb[ilevel+1][2]);
    restriction(mgb,ilevel,blockCoar,thread_a);
    exchangeField((*mgb.levels[ilevel+1]->bStorageRhs), 
                  (*mgb.levels[ilevel+1]->bStorageRhs_dev),
                  mgb.multilevel_bDecomp[ilevel+1],
                  (*mgb.levels[ilevel+1]->ev),
                  mgb.levels[ilevel+1]->timers.exchange_total);
    mgb.levels[ilevel+1]->num_operations.exchange_total+=1;
    squeue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class initXCoar>(sycl::nd_range<3>(blockCoar * thread_a, thread_a), 
            [=, grid = mgb.levels[ilevel+1]->grid_dev_ptr,
              bDat = mgb.levels[ilevel+1]->bStorageX_dev->dat.get(),
              bInfo = mgb.levels[ilevel+1]->bInfo_dev,
              stride = mgb.levels[ilevel+1]->grid_stride_dev]
            (sycl::nd_item<3> item) {
            auto bSize = cal_size<BDIM>::value;
            Brick_SYCL_3D bX(bInfo, bDat, bSize, 0);
            initX_kernel(item, grid, bX, stride);
        });
    });
  squeue.wait();
  }

  //Bottom solve
  int PR_iter = BOTTOM_SOLVER_ITER;
  int bottom_lvl = mgb.num_levels-1;
  dim3 block(mgb.multilevel_strideb[bottom_lvl][0], mgb.multilevel_strideb[bottom_lvl][1], mgb.multilevel_strideb[bottom_lvl][2]), 
       thread_a(TILE, TILE, TILE), thread(SYCL_SUBGROUP,1,1);
  //Point-relaxation
  exchange_pointRelax(mgb,bottom_lvl,block,thread,thread_a,PR_iter);

  for (int ilevel = mgb.num_levels-1; ilevel > start_lvl ; ilevel--)
  {
    int PR_iter_p = POST_SMOOTH_ITER;
    //Interpolation and Increment
    dim3 blockCoar(mgb.multilevel_strideb[ilevel][0], mgb.multilevel_strideb[ilevel][1], mgb.multilevel_strideb[ilevel][2]), 
    thread_a(TILE, TILE, TILE), thread(SYCL_SUBGROUP,1,1);
    interpolation_incr(mgb,ilevel,blockCoar,thread_a,1);
    dim3 block(mgb.multilevel_strideb[ilevel-1][0], mgb.multilevel_strideb[ilevel-1][1], mgb.multilevel_strideb[ilevel-1][2]); 
    //Point-relaxation
    exchange_pointRelax(mgb,ilevel-1,block,thread,thread_a,PR_iter_p);
  }

}

void initX_brick(MG_brick& mgb, dim3 block, dim3 thread_a){
  cl::sycl::queue squeue(*sycl_device, {cl::sycl::property::queue::enable_profiling()});

  for (int ilevel = 0; ilevel < mgb.num_levels-1; ilevel++)
  {
    dim3 block_a(mgb.multilevel_strideb[ilevel][0], mgb.multilevel_strideb[ilevel][1], mgb.multilevel_strideb[ilevel][2]);
    squeue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class initX>(sycl::nd_range<3>(block_a * thread_a, thread_a), 
            [=, grid = mgb.levels[ilevel]->grid_dev_ptr,
              bDat = mgb.levels[ilevel]->bStorageX_dev->dat.get(),
              bInfo = mgb.levels[ilevel]->bInfo_dev,
              stride = mgb.levels[ilevel]->grid_stride_dev]
            (sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(SYCL_SUBGROUP)]] {
            auto bSize = cal_size<BDIM>::value;
            Brick_SYCL_3D bX(bInfo, bDat, bSize, 0);
            initX_kernel(item, grid, bX, stride);
        });
    });
  }
};

void setTimersZero(MG_brick& mg){
  for (int ilevel = 0; ilevel < mg.num_levels; ilevel++){
    mg.levels[ilevel]->timers.pr = 0.0;
    mg.levels[ilevel]->num_operations.pr = 0.0;
    mg.levels[ilevel]->timers.apply_op = 0.0;
    mg.levels[ilevel]->num_operations.apply_op = 0.0;
    mg.levels[ilevel]->timers.restriction = 0.0;
    mg.levels[ilevel]->num_operations.restriction = 0.0;
    mg.levels[ilevel]->timers.interpolation_incr = 0.0;
    mg.levels[ilevel]->num_operations.interpolation_incr = 0.0;
    mg.levels[ilevel]->timers.exchange_total = 0.0;
    mg.levels[ilevel]->num_operations.exchange_total = 0.0;
    mg.levels[ilevel]->timers.maxNormRes = 0.0;
    mg.levels[ilevel]->num_operations.maxNormRes = 0.0;
  }
};

void maxNorm_brick(MG_brick& mgb, bElem *d_res){
  int level = 0;
  dim3 thread(BDIM);
  dim3 block(mgb.multilevel_strideb[level][0], mgb.multilevel_strideb[level][1], mgb.multilevel_strideb[level][2]);
  cl::sycl::queue squeue(*sycl_device, {cl::sycl::property::queue::enable_profiling()});
  squeue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<bElem> cache(sycl::range<1>(TILE*TILE*TILE), cgh); 
      cgh.parallel_for<class reduction>(sycl::nd_range<3>(block * thread, thread), 
          [=, grid = mgb.levels[level]->grid_dev_ptr,
            bDat = mgb.levels[level]->bStorageRes_dev->dat.get(),
            bInfo = mgb.levels[level]->bInfo_dev,
            stride = mgb.levels[level]->grid_stride_dev,
            res = d_res]
          (sycl::nd_item<3> item) {
          auto bSize = cal_size<BDIM>::value;
          Brick_SYCL_3D bRes(bInfo, bDat, bSize, 0);
          reduction_kernel(item, grid, bRes, stride, res, cache.get_multi_ptr<sycl::access::decorated::no>().get());
      });
  });

};
//------------------------------------------------------------------------------------------------------------------------------

