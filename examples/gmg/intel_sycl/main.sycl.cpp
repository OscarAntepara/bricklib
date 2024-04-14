
#include "stencils/fake.h"
#include <brick-mpi.h>
#include <brick.h>
#include <bricksetup.h>
#include <iostream>
#include <mpi.h>

#include "bitset.h"
#include <brickcompare.h>
#include <multiarray.h>

#include <array-mpi.h>
#include <unistd.h>

#include "args.h"
#include "mg_brick.h"
#include <fstream>
#include <iostream>


#undef BRICK_TOLERANCE
#define BRICK_TOLERANCE 1e-4

#define AMR_LEVELS 0

void print2DSlice_brick(Brick3D &out, std::vector<long> &strideb, unsigned *grid_ptr) {
  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::ofstream myfile;
  std::string fileName = "data2DAMR_brick.";
  std::string fileExtension = ".dat";
  std::string fullFileName = fileName + std::to_string(rank) + fileExtension;
  myfile.open(fullFileName);
  auto grid = (unsigned(*)[strideb[1]][strideb[0]])grid_ptr;
  int gb = 1;
  for (long jb = gb; jb < strideb[1] - gb; ++jb) {
    for (long j = 0; j < TILE; ++j) {
      myfile << "\n";
      for (long ib = gb; ib < strideb[0] - gb; ++ib) {
        unsigned b = grid[(strideb[2]/2)][jb][ib];
        for (long i = 0; i < TILE; ++i) {
          myfile << out[b][0][j][i] << " ";
        }
      }
    }
  }
  myfile.close();
}

template<typename T, typename mg>
double mgTime_mpi(T func, int &cnt, mg &a_mg) {
  int it = MPI_ITER;
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cnt = 0;
  if (rank == 0)
    std::cout << "Warm up" << std::endl;

  for (int i = 0; i < 10; ++i){
    if (rank == 0) std::cout << "Test # " <<i+1<< std::endl;
    func(); // Warm up
  }
  setTimersZero(a_mg);
  calctime = 0.0;
  packtime = calltime = waittime = movetime = calctime = 0;
  if (rank == 0)
    std::cout <<std::endl<< "Running solver " <<MPI_ITER<<" times for statistics"<<std::endl<<std::endl;

  double st = omp_get_wtime(), ed;
  for (int i = 0; i < MPI_ITER; ++i){
    if (rank == 0) std::cout << "Test # " <<i+1<< std::endl;
    func();
  }
  ed = omp_get_wtime();
  cnt = it;
  return (ed - st) / it;
}

void time_MgFunc(std::vector<mpi_stats>& func, int num_levels, const char* name){
  double total_func = 0.0;
  for (int ilevel = 0; ilevel < num_levels; ilevel++)
  {
    std::cout << "level "<<ilevel<< " "<<name<< " " << func[ilevel] << std::endl;
    total_func += func[ilevel].avg;
  }
  std::cout << "Total" << " " <<name<< " " << total_func << std::endl;
}

int main(int argc, char **argv) {
  MPI_ITER = 100;
  mg_num_lvls = 0;
  max_num_iter = 20;
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided != MPI_THREAD_SERIALIZED) {
    MPI_Finalize();
    return 1;
  }

  MPI_Comm cart = parseArgs(argc, argv, "sycl");

  if (cart != MPI_COMM_NULL) {

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MEMFD::setup_prefix("mpi-main", rank);

    int prd[3] = {1, 1, 1};
    int coo[3];
    MPI_Cart_get(cart, 3, (int *)dim_size.data(), prd, coo);

    int num_levels = mg_num_lvls+1;

    if (rank == 0){
      std::cout << "Running with MPI_GPU_AWARE = " << 0 << std::endl;
    }
    int cnt;
    double total;

    MG_brick mgb;
    init_mg_brick(mgb,num_levels,cart,coo,dom_size,AMR_LEVELS);

    size_t tsize = 0;
    for (auto &g : mgb.multilevel_bDecomp[0]->ghost)
      tsize += g.len * (*mgb.levels[0]->bStorageX).step * sizeof(bElem) * 2;

    int niterMG_brick = 0;

    auto brick_func = [&]() -> void {
      dim3 thread_a(BDIM);
      dim3 block(mgb.multilevel_strideb[0][0], mgb.multilevel_strideb[0][1], mgb.multilevel_strideb[0][2]);
      initX_brick(mgb, block, thread_a);

      niterMG_brick = 0;
      bElem resMG = 1.0;

      double st = omp_get_wtime(), ed;
      while (resMG > 1e-10 && niterMG_brick < max_num_iter){

        vcycle_brick(mgb,AMR_LEVELS);

        double elapsed=0.0;
        double st2 = omp_get_wtime();

        bElem *d_res;
        bElem h_res[1];
        h_res[0]=0.0;
        gpuMalloc(&d_res,sizeof(bElem));
        gpuMemcpy(d_res, h_res, sizeof(bElem), gpuMemcpyHostToDevice);

        maxNorm_brick(mgb,d_res);

        gpuMemcpy(h_res, d_res, sizeof(bElem), gpuMemcpyDeviceToHost);
        resMG = h_res[0];

        gpuFree(d_res);
        double max_allProc = 0.0;
        MPI_Reduce(&resMG, &max_allProc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        niterMG_brick++;

        MPI_Barrier(MPI_COMM_WORLD);
        elapsed = omp_get_wtime()-st2;
        mgb.levels[0]->timers.maxNormRes += elapsed;

        if (rank == 0) {
          printf("max resGPU %1.8e iter %d \n", max_allProc, niterMG_brick);
        }
      }
      ed = omp_get_wtime();
      calctime += (ed - st);
      if (rank == 0) printf("========================= \n");
    };

    total = mgTime_mpi(brick_func, cnt, mgb);

    {
      mpi_stats calc_s = mpi_statistics(calctime / cnt, MPI_COMM_WORLD);
      mpi_stats call_s = mpi_statistics(calltime / cnt, MPI_COMM_WORLD);
      mpi_stats wait_s = mpi_statistics(waittime / cnt, MPI_COMM_WORLD);
      mpi_stats mspd_s =
          mpi_statistics(tsize / 1.0e9 / (calltime + waittime) * cnt, MPI_COMM_WORLD);
      mpi_stats size_s = mpi_statistics((double)tsize * 1.0e-6, MPI_COMM_WORLD);
      mpi_stats move_s = mpi_statistics(movetime / cnt, MPI_COMM_WORLD);

      std::vector<mpi_stats> applyOp;
      std::vector<mpi_stats> pr;
      std::vector<mpi_stats> restr;
      std::vector<mpi_stats> interp;
      std::vector<mpi_stats> exchg;
      std::vector<mpi_stats> applyOp_nt;
      std::vector<mpi_stats> pr_nt;
      std::vector<mpi_stats> restr_nt;
      std::vector<mpi_stats> interp_nt;
      std::vector<mpi_stats> exchg_nt;
      for (int ilevel = 0; ilevel < num_levels; ilevel++)
      {
        mpi_stats applyOp_s = mpi_statistics(mgb.levels[ilevel]->timers.apply_op / cnt, MPI_COMM_WORLD);
        mpi_stats pr_s = mpi_statistics(mgb.levels[ilevel]->timers.pr / cnt, MPI_COMM_WORLD);
        mpi_stats restr_s = mpi_statistics(mgb.levels[ilevel]->timers.restriction / cnt, MPI_COMM_WORLD);
        mpi_stats interp_s = mpi_statistics(mgb.levels[ilevel]->timers.interpolation_incr / cnt, MPI_COMM_WORLD);
        mpi_stats exchg_s = mpi_statistics(mgb.levels[ilevel]->timers.exchange_total / cnt, MPI_COMM_WORLD);
        applyOp.push_back(applyOp_s);
        pr.push_back(pr_s);
        restr.push_back(restr_s);
        interp.push_back(interp_s);
        exchg.push_back(exchg_s);
        applyOp_nt.push_back(mpi_statistics(mgb.levels[ilevel]->num_operations.apply_op / cnt, MPI_COMM_WORLD));
        pr_nt.push_back(mpi_statistics(mgb.levels[ilevel]->num_operations.pr / cnt, MPI_COMM_WORLD));
        restr_nt.push_back(mpi_statistics(mgb.levels[ilevel]->num_operations.restriction / cnt, MPI_COMM_WORLD));
        interp_nt.push_back(mpi_statistics(mgb.levels[ilevel]->num_operations.interpolation_incr / cnt, MPI_COMM_WORLD));
        exchg_nt.push_back(mpi_statistics(mgb.levels[ilevel]->num_operations.exchange_total / cnt, MPI_COMM_WORLD));
      }
      mpi_stats maxNormRes_s = mpi_statistics(mgb.levels[0]->timers.maxNormRes / cnt, MPI_COMM_WORLD);


      if (rank == 0) {
        std::cout << "Total Time per operation and level [min,avg,max] (stdev)" << std::endl;
        time_MgFunc(pr, num_levels, "smooth+residual");
        time_MgFunc(applyOp, num_levels, "applyOp");
        time_MgFunc(restr, num_levels, "restriction");
        time_MgFunc(interp, num_levels, "interpolation+incr");
        time_MgFunc(exchg, num_levels, "exchange");
        std::cout << "maxNormRes " << maxNormRes_s << std::endl;
        std::cout << "======================== " << std::endl;
        std::cout << "Timings per invocation " << std::endl;
        double total_lvl = 0;
        for (int ilevel = 0; ilevel < num_levels; ilevel++)
        {
          std::cout << "level "<<ilevel<<" smooth+residual - time per inv: "<< pr[ilevel].avg/pr_nt[ilevel].avg << std::endl; 
          std::cout << "level "<<ilevel<<" applyOp - time per inv: "<< applyOp[ilevel].avg/applyOp_nt[ilevel].avg << std::endl; 
          std::cout << "level "<<ilevel<<" restriction - time per inv: "<< restr[ilevel].avg/restr_nt[ilevel].avg << std::endl; 
          std::cout << "level "<<ilevel<<" interpolation+incr - time per inv: "<< interp[ilevel].avg/interp_nt[ilevel].avg << std::endl; 
          std::cout << "level "<<ilevel<<" exchange - time per inv: "<< exchg[ilevel].avg/exchg_nt[ilevel].avg << std::endl; 
          total_lvl += pr[ilevel].avg + applyOp[ilevel].avg + 
                       restr[ilevel].avg + interp[ilevel].avg + 
                       exchg[ilevel].avg;
        }  
        std::cout << "======================== " << std::endl;
        std::cout << "Total Time per Level " << std::endl;
        for (int ilevel = 0; ilevel < num_levels; ilevel++)
        {
          std::cout << "level "<<ilevel<<" Total time "<< pr[ilevel].avg + applyOp[ilevel].avg + 
                                                          restr[ilevel].avg + interp[ilevel].avg +
                                                          + exchg[ilevel].avg  << std::endl;
        }      
        std::cout << "======================== " << std::endl;
        std::cout << "Bricks-GMG Total Time: " << total_lvl + maxNormRes_s.avg << std::endl;
        double perf = (double)tot_elems * 1.0e-9;
        perf = perf / (total_lvl + maxNormRes_s.avg);
        std::cout << "Perf " << perf << " GStencil/s" << std::endl;
      }
    }

    for (int ilevel = 0; ilevel < num_levels; ilevel++)
    {
      free(mgb.levels[ilevel]);
      free(mgb.levels[ilevel]->bInfo->adj);
      free(mgb.levels[ilevel]->grid_ptr);
      gpuFree(mgb.levels[ilevel]->bInfo_dev);
    }
    free(mgb.levels);


  }

  MPI_Finalize();
  return 0;
}
