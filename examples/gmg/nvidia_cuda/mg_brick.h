//------------------------------------------------------------------------------------------------------------------------------
#ifndef MG_BRICK_H
#define MG_BRICK_H
//------------------------------------------------------------------------------------------------------------------------------

#include "stencils/stencils.h"
#include <brick-cuda.h>
#include <brick-mpi.h>
#include <brick.h>
#include <bricksetup.h>

#include "stencils/cudaarray.h"
#include "stencils/cudavfold.h"
#include <multiarray.h>


#undef BRICK_TOLERANCE
#define BRICK_TOLERANCE 1e-4

#define AMR_LEVELS 0

#define VSVEC "CUDA"
#define WARPSIZE 32
#undef VFOLD
#define VFOLD 4, 8

#undef TILE
#define TILE 8

#undef PADDING
#define PADDING TILE

#undef GZ
#define GZ TILE

typedef Brick<Dim<BDIM>, Dim<VFOLD>> Brick3D;

struct levels_brick {
  bElem *x_ptr;
  bElem *Ax_ptr;
  bElem *rhs_ptr;
  bElem *res_ptr;
  BrickInfo<3>* bInfo;
  BrickInfo<3>* bInfo_dev;
  BrickStorage* bStorageX;
  BrickStorage* bStorageAx;
  BrickStorage* bStorageRhs;
  BrickStorage* bStorageRes;
  BrickStorage* bStorageX_dev;
  BrickStorage* bStorageAx_dev;
  BrickStorage* bStorageRhs_dev;
  BrickStorage* bStorageRes_dev;
  unsigned* grid_ptr;
  unsigned* dd_ptr; // Domain decomposition info 
  Brick3D* bX;
  Brick3D* bAx;
  Brick3D* bRhs;
  Brick3D* bRes;
  Brick3D* bX_dev;
  Brick3D* bAx_dev;
  Brick3D* bRhs_dev;
  Brick3D* bRes_dev;
  bElem* dom_len_dev;
  unsigned *grid_dev_ptr;
  unsigned* dd_dev_ptr;
  unsigned *grid_stride_dev;
  ExchangeView* ev;
  // statistics information...
  struct {
    double                  pr;
    double            apply_op;
    double         restriction;
    double  interpolation_incr;
    double      exchange_total;
    double          maxNormRes;
  }timers;
  struct {
    int                  pr=0;
    int            apply_op=0;
    int         restriction=0;
    int  interpolation_incr=0;
    int      exchange_total=0;
    int          maxNormRes=0;
  }num_operations;
};

struct MG_brick {
  int       num_levels; // depth of the v-cycle
  levels_brick ** levels; // array of pointers to levels data
  std::vector<std::vector<long> > multilevel_stride;
  std::vector<std::vector<long> > multilevel_strideb;
  std::vector<std::vector<long> > multilevel_strideg;
  std::vector<std::shared_ptr<BrickDecomp<3, BDIM> >> multilevel_bDecomp;
  int       mpi_size;
};


void init_mg_brick(MG_brick& mg, int num_levels, MPI_Comm cart,int* coo, std::vector<unsigned>& dom_size, int amr_levels);
void vcycle_brick(MG_brick& mgb, int start_lvl);

void initX_brick(MG_brick& mgb, dim3 block, dim3 thread_a);
void setTimersZero(MG_brick& mg);
void maxNorm_brick(MG_brick& mg, bElem *d_res);
//------------------------------------------------------------------------------------------------------------------------------
#endif

