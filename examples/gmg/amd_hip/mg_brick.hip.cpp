//------------------------------------------------------------------------------------------------------------------------------

#include "mg_brick.h"
#include "mg_common.h"

#define WARP_SIZE 64

//7pt stencil coefficients for Poisson Op
#undef MPI_ALPHA
#undef MPI_BETA
#define MPI_ALPHA -6
#define MPI_BETA 1

//Python file name for applyOp using code-gen
#undef ST_SCRTPT
#define ST_SCRTPT "mpi7pt.py"

//Relaxation coefficient for GSRB and Jacobi 
#define JACOBI_COEFF 1.0/9.0

//Number of smooths going down and up in the v-cycle, as well as, number of iterations at the bottom solver
#define PRE_SMOOTH_ITER 6
#define POST_SMOOTH_ITER 6
#define BOTTOM_SOLVER_ITER 100


__device__ double 
atomicMaxB(bElem* address, bElem val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(max(val,
                      __longlong_as_double(assumed))));
  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}

__global__ void 
reduction_kernel(unsigned *grid, Brick3D res, unsigned *stride, bElem *d_res)
{
  __shared__ bElem cache[TILE*TILE*TILE];
  // accumulate per thread first (multiple elements)
  bElem thread_val = 0.0;

  unsigned bk = 1 + blockIdx.z;
  unsigned bj = 1 + blockIdx.y;
  unsigned bi = 1 + blockIdx.x;
  long k = threadIdx.z;
  long j = threadIdx.y;
  long i = threadIdx.x;
  unsigned b = grid[bi + (bj + bk * stride[1]) * stride[0]];

  if (bi < (stride[1] - 1) && bj < (stride[1] - 1) && bk < (stride[1] - 1)) {
    double val = res[b][k][j][i];
    thread_val = max(thread_val, fabs(val));
  }

  int cacheIndex = threadIdx.x + threadIdx.y*TILE + threadIdx.z*TILE*TILE;
  cache[cacheIndex] = thread_val;   // set the cache value 

  __syncthreads();

  // perform parallel reduction, threadsPerBlock must be 2^m

  int ib = (blockDim.x*blockDim.y*blockDim.z) / 2;
  while (ib != 0) {
    if(cacheIndex < ib && cache[cacheIndex + ib] > cache[cacheIndex])
      cache[cacheIndex] = cache[cacheIndex + ib]; 

    __syncthreads();

    ib /= 2;
  }

  if(cacheIndex == 0) atomicMaxB(d_res, cache[0]);
}

__global__ void 
initX_kernel(unsigned *grid, Brick3D x, unsigned *stride) {
  unsigned bk = blockIdx.z;
  unsigned bj = blockIdx.y;
  unsigned bi = blockIdx.x;
  long k = threadIdx.z;
  long j = threadIdx.y;
  long i = threadIdx.x;
  unsigned b = grid[bi + (bj + bk * stride[1]) * stride[0]];

  x[b][k][j][i] = 0.0;
}

void exchangeField(BrickStorage& field,
                   BrickStorage& field_dev,
                   std::shared_ptr<BrickDecomp<3, BDIM> >& bDecomp,
                   ExchangeView& ev,
                   int mpi_size,
                   Brick3D& field_brick,
                   unsigned* dd_dev,
                   double& timer)
{

  double st1 = omp_get_wtime();
  hipDeviceSynchronize();
#ifndef GPU_AWARE
  {
    double t_a = omp_get_wtime();
    hipMemcpy(field.dat.get() + field.step * bDecomp->sep_pos[0],
                field_dev.dat.get() + field.step * bDecomp->sep_pos[0],
                field.step * (bDecomp->sep_pos[1] - bDecomp->sep_pos[0]) * sizeof(bElem),
                hipMemcpyDeviceToHost);
    double t_b = omp_get_wtime();
    movetime += t_b - t_a;
#ifdef DECOMP_PAGEUNALIGN
    bDecomp->exchange(field);
#else
    //(*mgb.levels[ilevel+1]->ev).exchange();
#endif
    t_a = omp_get_wtime();
    hipMemcpy(field_dev.dat.get() + field.step * bDecomp->sep_pos[1],
                field.dat.get() + field.step * bDecomp->sep_pos[1],
                field.step * (bDecomp->sep_pos[2] - bDecomp->sep_pos[1]) * sizeof(bElem),
                hipMemcpyHostToDevice);
    t_b = omp_get_wtime();
    movetime += t_b - t_a;
  }
#else
    bDecomp->exchange(field_dev);
#endif
  
  timer+= calltime + waittime;
  calltime=0.0;
  waittime=0.0;

}

__global__ void 
applyOp_kernel(unsigned *grid, Brick3D x, Brick3D Ax, unsigned *stride) {
  unsigned bk = blockIdx.z;
  unsigned bj = blockIdx.y;
  unsigned bi = blockIdx.x;

  unsigned b = grid[bi + (bj + bk * stride[1]) * stride[0]];

  brick(ST_SCRTPT, VSVEC, (BDIM), (VFOLD), b);
}

__global__ void 
smooth_residual_kernel(unsigned *grid, Brick3D x, Brick3D Ax, Brick3D rhs, Brick3D res,
                       bElem* dom_len_dev, unsigned *stride) {
  unsigned bk = blockIdx.z;
  unsigned bj = blockIdx.y;
  unsigned bi = blockIdx.x;
  long k = threadIdx.z;
  long j = threadIdx.y;
  long i = threadIdx.x;
  unsigned b = grid[bi + (bj + bk * stride[1]) * stride[0]];
  
  x[b][k][j][i] += JACOBI_COEFF * (Ax[b][k][j][i] - (dom_len_dev[0] * rhs[b][k][j][i]));
  res[b][k][j][i] = rhs[b][k][j][i] - (Ax[b][k][j][i]*dom_len_dev[1]);
}


__global__ void 
smooth_kernel(unsigned *grid, Brick3D x, Brick3D Ax, Brick3D rhs, Brick3D res,
                       bElem* dom_len_dev, unsigned *stride) {
  unsigned bk = blockIdx.z;
  unsigned bj = blockIdx.y;
  unsigned bi = blockIdx.x;
  long k = threadIdx.z;
  long j = threadIdx.y;
  long i = threadIdx.x;
  unsigned b = grid[bi + (bj + bk * stride[1]) * stride[0]];
  
  x[b][k][j][i] += JACOBI_COEFF * (Ax[b][k][j][i] - (dom_len_dev[0] * rhs[b][k][j][i]));
}

__global__ void
restriction_kernel(unsigned *grid, unsigned *gridCoar,
                 Brick3D res, Brick3D rhs, unsigned *stride, unsigned *strideCoar) {

  unsigned bk = 1 + blockIdx.z;
  unsigned bj = 1 + blockIdx.y;
  unsigned bi = 1 + blockIdx.x;
  unsigned bFk = 1 + blockIdx.z*2;
  unsigned bFj = 1 + blockIdx.y*2;
  unsigned bFi = 1 + blockIdx.x*2;
  long k = threadIdx.z;
  long j = threadIdx.y;
  long i = threadIdx.x;
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


__global__ void
interpolation_incr_kernel(unsigned *grid, unsigned *gridCoar,
                         Brick3D xCoar, Brick3D x, unsigned *stride, unsigned *strideCoar) {

  unsigned bk = 1 + blockIdx.z;
  unsigned bj = 1 + blockIdx.y;
  unsigned bi = 1 + blockIdx.x;
  unsigned bFk = 1 + blockIdx.z*2;
  unsigned bFj = 1 + blockIdx.y*2;
  unsigned bFi = 1 + blockIdx.x*2;
  long k = threadIdx.z;
  long j = threadIdx.y;
  long i = threadIdx.x;
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
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  mg.mpi_size = size;

  mg.num_levels = num_levels;
  mg.levels = (levels_brick**)malloc(num_levels*sizeof(levels_brick*));

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
    auto _bInfo_dev = movBrickInfo((*mg.levels[ilevel]->bInfo), hipMemcpyHostToDevice);
    {
      unsigned size = sizeof(BrickInfo<3>);
      hipMalloc(&mg.levels[ilevel]->bInfo_dev, size);
      hipMemcpy(mg.levels[ilevel]->bInfo_dev, &_bInfo_dev, size, hipMemcpyHostToDevice);
    }

    mg.levels[ilevel]->bStorageX_dev = new BrickStorage (movBrickStorage((*mg.levels[ilevel]->bStorageX), hipMemcpyHostToDevice));
    mg.levels[ilevel]->bStorageAx_dev = new BrickStorage (movBrickStorage((*mg.levels[ilevel]->bStorageAx), hipMemcpyHostToDevice));
    mg.levels[ilevel]->bStorageRhs_dev = new BrickStorage (movBrickStorage((*mg.levels[ilevel]->bStorageRhs), hipMemcpyHostToDevice));
    mg.levels[ilevel]->bStorageRes_dev = new BrickStorage (movBrickStorage((*mg.levels[ilevel]->bStorageRes), hipMemcpyHostToDevice));

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
    hipMalloc((void **) &mg.levels[ilevel]->dd_dev_ptr, ngBricksBytes);
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
    hipMemcpy(mg.levels[ilevel]->dd_dev_ptr, mg.levels[ilevel]->dd_ptr, ngBricksBytes, hipMemcpyHostToDevice);

    bElem dom_len_ptr[2];
    dom_len_ptr[0] = 1.0;
    mg.levels[ilevel]->dom_len_dev = nullptr;
    bElem dom_size = TILE * (mg.multilevel_strideb[ilevel][0] - 2);
    bElem h2 = dom_len_ptr[0]*dom_len_ptr[0]/(dom_size*dom_size);

    dom_len_ptr[0] = h2;
    dom_len_ptr[1] = 1.0/h2;
    hipMalloc((void**)&mg.levels[ilevel]->dom_len_dev,2*sizeof(bElem));
    hipMemcpy(mg.levels[ilevel]->dom_len_dev,dom_len_ptr,2*sizeof(bElem),hipMemcpyHostToDevice);
    

#ifndef DECOMP_PAGEUNALIGN
    mg.levels[ilevel]->ev = new ExchangeView (bDecomp->exchangeView((*mg.levels[ilevel]->bStorageX)));
#endif

  }
}

void point_relax(MG_brick& mgb, int level, dim3 block, dim3 thread, dim3 thread_a, int n_iter, int max_iter){
  float elapsed;
  hipEvent_t c_0, c_1;
  hipEventCreate(&c_1);
  hipEventCreate(&c_0);
  hipEventRecord(c_0);
  applyOp_kernel<<<block, thread>>>(mgb.levels[level]->grid_dev_ptr, (*mgb.levels[level]->bX_dev), (*mgb.levels[level]->bAx_dev), mgb.levels[level]->grid_stride_dev);
  hipEventRecord(c_1);
  hipEventSynchronize(c_1);
  hipEventElapsedTime(&elapsed, c_0, c_1);
  mgb.levels[level]->timers.apply_op += elapsed / 1000.0;
  mgb.levels[level]->num_operations.apply_op += 1;
  hipEventRecord(c_0);
  if (n_iter==max_iter-1){                                                 
    smooth_residual_kernel<<<block, thread_a>>>(mgb.levels[level]->grid_dev_ptr, (*mgb.levels[level]->bX_dev), (*mgb.levels[level]->bAx_dev), (*mgb.levels[level]->bRhs_dev), (*mgb.levels[level]->bRes_dev), 
                                           mgb.levels[level]->dom_len_dev, mgb.levels[level]->grid_stride_dev);
  }else{
    smooth_kernel<<<block, thread_a>>>(mgb.levels[level]->grid_dev_ptr, (*mgb.levels[level]->bX_dev), (*mgb.levels[level]->bAx_dev), (*mgb.levels[level]->bRhs_dev), (*mgb.levels[level]->bRes_dev), 
                                           mgb.levels[level]->dom_len_dev, mgb.levels[level]->grid_stride_dev);
  }    
  hipEventRecord(c_1);
  hipEventSynchronize(c_1);
  hipEventElapsedTime(&elapsed, c_0, c_1);
  mgb.levels[level]->timers.pr += elapsed / 1000.0;
  mgb.levels[level]->num_operations.pr += 1;
};

void restriction(MG_brick& mgb, int level, dim3 block, dim3 thread){
  float elapsed;
  hipEvent_t c_0, c_1;
  hipEventCreate(&c_1);
  hipEventCreate(&c_0);
  hipEventRecord(c_0);
  restriction_kernel <<< block, thread >>> (mgb.levels[level]->grid_dev_ptr, mgb.levels[level+1]->grid_dev_ptr, (*mgb.levels[level]->bRes_dev), (*mgb.levels[level+1]->bRhs_dev), 
                                            mgb.levels[level]->grid_stride_dev, mgb.levels[level+1]->grid_stride_dev);
  hipEventRecord(c_1);
  hipEventSynchronize(c_1);
  hipEventElapsedTime(&elapsed, c_0, c_1);
  mgb.levels[level]->timers.restriction += elapsed / 1000.0;
  mgb.levels[level]->num_operations.restriction += 1;
};

void interpolation_incr(MG_brick& mgb, int level, dim3 block, dim3 thread, int interp_type){
  float elapsed;
  hipEvent_t c_0, c_1;
  hipEventCreate(&c_1);
  hipEventCreate(&c_0);
  hipEventRecord(c_0);
  interpolation_incr_kernel <<< block, thread >>> (mgb.levels[level-1]->grid_dev_ptr, mgb.levels[level]->grid_dev_ptr, (*mgb.levels[level]->bX_dev), (*mgb.levels[level-1]->bX_dev), 
                                          mgb.levels[level-1]->grid_stride_dev, mgb.levels[level]->grid_stride_dev);
  hipEventRecord(c_1);
  hipEventSynchronize(c_1);
  hipEventElapsedTime(&elapsed, c_0, c_1);
  mgb.levels[level]->timers.interpolation_incr += elapsed / 1000.0;
  mgb.levels[level]->num_operations.interpolation_incr += 1;
};

void exchange_pointRelax(MG_brick& mgb, int level, dim3 block, dim3 thread, dim3 thread_a, int n_iter){
      for (int i = 0; i < n_iter; ++i) {
        //hipDeviceSynchronize();
        //hipEventCreate(&c_0);
        int n_avoidComm = 7;
if (i%n_avoidComm==0){
        exchangeField((*mgb.levels[level]->bStorageX), 
                      (*mgb.levels[level]->bStorageX_dev),
                      mgb.multilevel_bDecomp[level],
                      (*mgb.levels[level]->ev),
                      mgb.mpi_size,
                      (*mgb.levels[level]->bX_dev),
                      mgb.levels[level]->dd_dev_ptr,
                      mgb.levels[level]->timers.exchange_total);
        mgb.levels[level]->num_operations.exchange_total+=1;
}
        //why do we need 2 syncs? This works but further investigation is needed.
        //hipDeviceSynchronize();
        if (n_iter == BOTTOM_SOLVER_ITER){
          point_relax(mgb,level,block,thread,thread_a,i,n_iter);
        }else{
          point_relax(mgb,level,block,thread,thread_a,i,n_iter);
        }
        //hipDeviceSynchronize();

      }
};


void vcycle_brick(MG_brick& mgb, int start_lvl){

  for (int ilevel = start_lvl; ilevel < mgb.num_levels-1; ilevel++)
  {
    int PR_iter_r = PRE_SMOOTH_ITER;
    dim3 thread_a(BDIM);
    dim3 block(mgb.multilevel_strideb[ilevel][0], mgb.multilevel_strideb[ilevel][1], mgb.multilevel_strideb[ilevel][2]), 
         thread(WARP_SIZE);
    //Point-relaxation
    exchange_pointRelax(mgb,ilevel,block,thread,thread_a,PR_iter_r);
    //Restriction
    dim3 blockCoar(mgb.multilevel_strideb[ilevel+1][0], mgb.multilevel_strideb[ilevel+1][1], mgb.multilevel_strideb[ilevel+1][2]);
    restriction(mgb,ilevel,blockCoar,thread_a);
    exchangeField((*mgb.levels[ilevel+1]->bStorageRhs), 
                  (*mgb.levels[ilevel+1]->bStorageRhs_dev),
                  mgb.multilevel_bDecomp[ilevel+1],
                  (*mgb.levels[ilevel+1]->ev),
                  mgb.mpi_size,
                  (*mgb.levels[ilevel+1]->bRhs_dev),
                  mgb.levels[ilevel+1]->dd_dev_ptr,
                  mgb.levels[ilevel+1]->timers.exchange_total);
    mgb.levels[ilevel+1]->num_operations.exchange_total+=1;
    initX_kernel<<<blockCoar, thread_a>>>(mgb.levels[ilevel+1]->grid_dev_ptr, (*mgb.levels[ilevel+1]->bX_dev), mgb.levels[ilevel+1]->grid_stride_dev);
  }

  //Bottom solve
  int PR_iter = BOTTOM_SOLVER_ITER;
  int bottom_lvl = mgb.num_levels-1;
  dim3 block(mgb.multilevel_strideb[bottom_lvl][0], mgb.multilevel_strideb[bottom_lvl][1], mgb.multilevel_strideb[bottom_lvl][2]), 
       thread_a(TILE, TILE, TILE), thread(WARP_SIZE);
  //Point-relaxation
  exchange_pointRelax(mgb,bottom_lvl,block,thread,thread_a,PR_iter);

  for (int ilevel = mgb.num_levels-1; ilevel > start_lvl ; ilevel--)
  {
    int PR_iter_p = POST_SMOOTH_ITER;
    //Interpolation and Increment
    dim3 blockCoar(mgb.multilevel_strideb[ilevel][0], mgb.multilevel_strideb[ilevel][1], mgb.multilevel_strideb[ilevel][2]), 
    thread_a(TILE, TILE, TILE), thread(WARP_SIZE);
    interpolation_incr(mgb,ilevel,blockCoar,thread_a,1);
    dim3 block(mgb.multilevel_strideb[ilevel-1][0], mgb.multilevel_strideb[ilevel-1][1], mgb.multilevel_strideb[ilevel-1][2]); 
    //Point-relaxation
    exchange_pointRelax(mgb,ilevel-1,block,thread,thread_a,PR_iter_p);
  }

}

void initX_brick(MG_brick& mgb, dim3 block, dim3 thread_a){
  for (int ilevel = 0; ilevel < mgb.num_levels-1; ilevel++)
  {
    dim3 block_a(mgb.multilevel_strideb[ilevel][0], mgb.multilevel_strideb[ilevel][1], mgb.multilevel_strideb[ilevel][2]);
    initX_kernel<<<block_a, thread_a>>>(mgb.levels[ilevel]->grid_dev_ptr, (*mgb.levels[ilevel]->bX_dev), 
                                        mgb.levels[ilevel]->grid_stride_dev);
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
  reduction_kernel<<<block, thread>>>(mgb.levels[level]->grid_dev_ptr, (*mgb.levels[level]->bRes_dev), mgb.levels[level]->grid_stride_dev, d_res);
};


//------------------------------------------------------------------------------------------------------------------------------

