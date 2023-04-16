#ifndef BRICK_MACROS_COEFFS_H
#define BRICK_MACROS_COEFFS_H

#include "brick.h"

//STAR_STENCIL_RADIUS-valid values are 1,2,3,4
//CUBE_STENCIL_RADIUS-valid values are 1,2
#ifdef STENCIL_RADIUS_1
#define STAR_STENCIL_RADIUS 1
#define CUBE_STENCIL_RADIUS 1
#elif STENCIL_RADIUS_2
#define STAR_STENCIL_RADIUS 2
#define CUBE_STENCIL_RADIUS 2
#elif STENCIL_RADIUS_3
#define STAR_STENCIL_RADIUS 3
#define CUBE_STENCIL_RADIUS 1
#elif STENCIL_RADIUS_4
#define STAR_STENCIL_RADIUS 4
#define CUBE_STENCIL_RADIUS 1
#endif

#define MPI_BETA 0.1
#define MPI_ALPHA 0.4

#define MPI_A0 0.1
#define MPI_A1 0.06
#define MPI_A2 0.045
#define MPI_A3 0.03
#define MPI_A4 0.015

#define MPI_B0 0.4
#define MPI_B1 0.07
#define MPI_B2 0.03

#define MPI_C0 0.1
#define MPI_C1 0.04
#define MPI_C2 0.03
#define MPI_C3 0.01
#define MPI_C4 0.006
#define MPI_C5 0.004
#define MPI_C6 0.005
#define MPI_C7 0.002
#define MPI_C8 0.003
#define MPI_C9 0.001

#ifdef STAR_STENCIL_RADIUS
#if STAR_STENCIL_RADIUS==1
#define STAR_7PT
#elif STAR_STENCIL_RADIUS==2
#define STAR_13PT
#elif STAR_STENCIL_RADIUS==3
#define STAR_19PT
#elif STAR_STENCIL_RADIUS==4
#define STAR_25PT
#endif
#endif 

#ifdef CUBE_STENCIL_RADIUS
#if CUBE_STENCIL_RADIUS==1
#define CUBE_27PT
#elif CUBE_STENCIL_RADIUS==2
#define CUBE_125PT
#endif
#endif 

#ifdef STAR_7PT
#define ST_STAR_CPU arr_out[k][j][i] = (arr_in[k + 1][j][i] + arr_in[k - 1][j][i] + \
                                  arr_in[k][j + 1][i] + arr_in[k][j - 1][i] + \
                                  arr_in[k][j][i + 1] + arr_in[k][j][i - 1]) * MPI_BETA + \
                                 arr_in[k][j][i] * MPI_ALPHA
#define ST_STAR_ARR_GPU arr_out[k][j][i] = (arr_in[k + 1][j][i] + arr_in[k - 1][j][i] + \
                                  arr_in[k][j + 1][i] + arr_in[k][j - 1][i] + \
                                  arr_in[k][j][i + 1] + arr_in[k][j][i - 1]) * MPI_BETA + \
                                 arr_in[k][j][i] * MPI_ALPHA
#define ST_STAR_BRICK_GPU bOut[b][k][j][i] = (bIn[b][k + 1][j][i] + bIn[b][k - 1][j][i] + \
                                  bIn[b][k][j + 1][i] + bIn[b][k][j - 1][i] + \
                                  bIn[b][k][j][i + 1] + bIn[b][k][j][i - 1]) * MPI_BETA + \
                                 bIn[b][k][j][i] * MPI_ALPHA
#elif defined(STAR_13PT)
#define ST_STAR_CPU arr_out[k][j][i] = (arr_in[k + 2][j][i] + arr_in[k - 2][j][i] + \
                                  arr_in[k][j + 2][i] + arr_in[k][j - 2][i] + \
                                  arr_in[k][j][i + 2] + arr_in[k][j][i - 2]) * MPI_B2 + \
                                 (arr_in[k + 1][j][i] + arr_in[k - 1][j][i] + \
                                  arr_in[k][j + 1][i] + arr_in[k][j - 1][i] + \
                                  arr_in[k][j][i + 1] + arr_in[k][j][i - 1]) * MPI_B1 + \
                                 arr_in[k][j][i] * MPI_B0
#define ST_STAR_ARR_GPU arr_out[k][j][i] = (arr_in[k + 2][j][i] + arr_in[k - 2][j][i] + \
                                  arr_in[k][j + 2][i] + arr_in[k][j - 2][i] + \
                                  arr_in[k][j][i + 2] + arr_in[k][j][i - 2]) * MPI_B2 + \
                                 (arr_in[k + 1][j][i] + arr_in[k - 1][j][i] + \
                                  arr_in[k][j + 1][i] + arr_in[k][j - 1][i] + \
                                  arr_in[k][j][i + 1] + arr_in[k][j][i - 1]) * MPI_B1 + \
                                 arr_in[k][j][i] * MPI_B0
#define ST_STAR_BRICK_GPU bOut[b][k][j][i] = (bIn[b][k + 2][j][i] + bIn[b][k - 2][j][i] + \
                                  bIn[b][k][j + 2][i] + bIn[b][k][j - 2][i] + \
                                  bIn[b][k][j][i + 2] + bIn[b][k][j][i - 2]) * MPI_B2 + \
                                 (bIn[b][k + 1][j][i] + bIn[b][k - 1][j][i] + \
                                  bIn[b][k][j + 1][i] + bIn[b][k][j - 1][i] + \
                                  bIn[b][k][j][i + 1] + bIn[b][k][j][i - 1]) * MPI_B1 + \
                                 bIn[b][k][j][i] * MPI_B0
#elif defined(STAR_19PT)
#define ST_STAR_CPU arr_out[k][j][i] = (arr_in[k + 3][j][i] + arr_in[k - 3][j][i] + \
                                  arr_in[k][j + 3][i] + arr_in[k][j - 3][i] + \
                                  arr_in[k][j][i + 3] + arr_in[k][j][i - 3]) * MPI_A3 + \
                                 (arr_in[k + 2][j][i] + arr_in[k - 2][j][i] + \
                                  arr_in[k][j + 2][i] + arr_in[k][j - 2][i] + \
                                  arr_in[k][j][i + 2] + arr_in[k][j][i - 2]) * MPI_A2 + \
                                 (arr_in[k + 1][j][i] + arr_in[k - 1][j][i] + \
                                  arr_in[k][j + 1][i] + arr_in[k][j - 1][i] + \
                                  arr_in[k][j][i + 1] + arr_in[k][j][i - 1]) * MPI_A1 + \
                                 arr_in[k][j][i] * MPI_A0
#define ST_STAR_ARR_GPU arr_out[k][j][i] = (arr_in[k + 3][j][i] + arr_in[k - 3][j][i] + \
                                  arr_in[k][j + 3][i] + arr_in[k][j - 3][i] + \
                                  arr_in[k][j][i + 3] + arr_in[k][j][i - 3]) * MPI_A3 + \
                                 (arr_in[k + 2][j][i] + arr_in[k - 2][j][i] + \
                                  arr_in[k][j + 2][i] + arr_in[k][j - 2][i] + \
                                  arr_in[k][j][i + 2] + arr_in[k][j][i - 2]) * MPI_A2 + \
                                 (arr_in[k + 1][j][i] + arr_in[k - 1][j][i] + \
                                  arr_in[k][j + 1][i] + arr_in[k][j - 1][i] + \
                                  arr_in[k][j][i + 1] + arr_in[k][j][i - 1]) * MPI_A1 + \
                                 arr_in[k][j][i] * MPI_A0
#define ST_STAR_BRICK_GPU bOut[b][k][j][i] = (bIn[b][k + 3][j][i] + bIn[b][k - 3][j][i] + \
                                  bIn[b][k][j + 3][i] + bIn[b][k][j - 3][i] + \
                                  bIn[b][k][j][i + 3] + bIn[b][k][j][i - 3]) * MPI_A3 + \
                                 (bIn[b][k + 2][j][i] + bIn[b][k - 2][j][i] + \
                                  bIn[b][k][j + 2][i] + bIn[b][k][j - 2][i] + \
                                  bIn[b][k][j][i + 2] + bIn[b][k][j][i - 2]) * MPI_A2 + \
                                 (bIn[b][k + 1][j][i] + bIn[b][k - 1][j][i] + \
                                  bIn[b][k][j + 1][i] + bIn[b][k][j - 1][i] + \
                                  bIn[b][k][j][i + 1] + bIn[b][k][j][i - 1]) * MPI_A1 + \
                                 bIn[b][k][j][i] * MPI_A0
#elif defined(STAR_25PT)
#define ST_STAR_CPU arr_out[k][j][i] = (arr_in[k + 4][j][i] + arr_in[k - 4][j][i] + \
                                  arr_in[k][j + 4][i] + arr_in[k][j - 4][i] + \
                                  arr_in[k][j][i + 4] + arr_in[k][j][i - 4]) * MPI_A4 + \
                                 (arr_in[k + 3][j][i] + arr_in[k - 3][j][i] + \
                                  arr_in[k][j + 3][i] + arr_in[k][j - 3][i] + \
                                  arr_in[k][j][i + 3] + arr_in[k][j][i - 3]) * MPI_A3 + \
                                 (arr_in[k + 2][j][i] + arr_in[k - 2][j][i] + \
                                  arr_in[k][j + 2][i] + arr_in[k][j - 2][i] + \
                                  arr_in[k][j][i + 2] + arr_in[k][j][i - 2]) * MPI_A2 + \
                                 (arr_in[k + 1][j][i] + arr_in[k - 1][j][i] + \
                                  arr_in[k][j + 1][i] + arr_in[k][j - 1][i] + \
                                  arr_in[k][j][i + 1] + arr_in[k][j][i - 1]) * MPI_A1 + \
                                 arr_in[k][j][i] * MPI_A0
#define ST_STAR_ARR_GPU arr_out[k][j][i] = (arr_in[k + 4][j][i] + arr_in[k - 4][j][i] + \
                                  arr_in[k][j + 4][i] + arr_in[k][j - 4][i] + \
                                  arr_in[k][j][i + 4] + arr_in[k][j][i - 4]) * MPI_A4 + \
                                 (arr_in[k + 3][j][i] + arr_in[k - 3][j][i] + \
                                  arr_in[k][j + 3][i] + arr_in[k][j - 3][i] + \
                                  arr_in[k][j][i + 3] + arr_in[k][j][i - 3]) * MPI_A3 + \
                                 (arr_in[k + 2][j][i] + arr_in[k - 2][j][i] + \
                                  arr_in[k][j + 2][i] + arr_in[k][j - 2][i] + \
                                  arr_in[k][j][i + 2] + arr_in[k][j][i - 2]) * MPI_A2 + \
                                 (arr_in[k + 1][j][i] + arr_in[k - 1][j][i] + \
                                  arr_in[k][j + 1][i] + arr_in[k][j - 1][i] + \
                                  arr_in[k][j][i + 1] + arr_in[k][j][i - 1]) * MPI_A1 + \
                                 arr_in[k][j][i] * MPI_A0
#define ST_STAR_BRICK_GPU bOut[b][k][j][i] = (bIn[b][k + 4][j][i] + bIn[b][k - 4][j][i] + \
                                  bIn[b][k][j + 4][i] + bIn[b][k][j - 4][i] + \
                                  bIn[b][k][j][i + 4] + bIn[b][k][j][i - 4]) * MPI_A4 + \
                                 (bIn[b][k + 3][j][i] + bIn[b][k - 3][j][i] + \
                                  bIn[b][k][j + 3][i] + bIn[b][k][j - 3][i] + \
                                  bIn[b][k][j][i + 3] + bIn[b][k][j][i - 3]) * MPI_A3 + \
                                 (bIn[b][k + 2][j][i] + bIn[b][k - 2][j][i] + \
                                  bIn[b][k][j + 2][i] + bIn[b][k][j - 2][i] + \
                                  bIn[b][k][j][i + 2] + bIn[b][k][j][i - 2]) * MPI_A2 + \
                                 (bIn[b][k + 1][j][i] + bIn[b][k - 1][j][i] + \
                                  bIn[b][k][j + 1][i] + bIn[b][k][j - 1][i] + \
                                  bIn[b][k][j][i + 1] + bIn[b][k][j][i - 1]) * MPI_A1 + \
                                 bIn[b][k][j][i] * MPI_A0
#endif

#if defined(CUBE_27PT)
#define ST_CUBE_CPU arr_out[k][j][i] = ( \
       MPI_C0 * arr_in[k][j][i] + \
       MPI_C1 * (arr_in[k + 1][j][i] + \
                 arr_in[k - 1][j][i] + \
                 arr_in[k][j + 1][i] + \
                 arr_in[k][j - 1][i] + \
                 arr_in[k][j][i + 1] + \
                 arr_in[k][j][i - 1]) + \
       MPI_C3 * (arr_in[k + 1][j + 1][i] + \
                 arr_in[k - 1][j + 1][i] + \
                 arr_in[k + 1][j - 1][i] + \
                 arr_in[k - 1][j - 1][i] + \
                 arr_in[k + 1][j][i + 1] + \
                 arr_in[k - 1][j][i + 1] + \
                 arr_in[k + 1][j][i - 1] + \
                 arr_in[k - 1][j][i - 1] + \
                 arr_in[k][j + 1][i + 1] + \
                 arr_in[k][j - 1][i + 1] + \
                 arr_in[k][j + 1][i - 1] + \
                 arr_in[k][j - 1][i - 1]) + \
       MPI_C6 * (arr_in[k + 1][j + 1][i + 1] + \
                 arr_in[k - 1][j + 1][i + 1] + \
                 arr_in[k + 1][j - 1][i + 1] + \
                 arr_in[k - 1][j - 1][i + 1] + \
                 arr_in[k + 1][j + 1][i - 1] + \
                 arr_in[k - 1][j + 1][i - 1] + \
                 arr_in[k + 1][j - 1][i - 1] + \
                 arr_in[k - 1][j - 1][i - 1]) )
#define ST_CUBE_ARR_GPU arr_out[k][j][i] = ( \
       MPI_C0 * arr_in[k][j][i] + \
       MPI_C1 * (arr_in[k + 1][j][i] + \
                 arr_in[k - 1][j][i] + \
                 arr_in[k][j + 1][i] + \
                 arr_in[k][j - 1][i] + \
                 arr_in[k][j][i + 1] + \
                 arr_in[k][j][i - 1]) + \
       MPI_C3 * (arr_in[k + 1][j + 1][i] + \
                 arr_in[k - 1][j + 1][i] + \
                 arr_in[k + 1][j - 1][i] + \
                 arr_in[k - 1][j - 1][i] + \
                 arr_in[k + 1][j][i + 1] + \
                 arr_in[k - 1][j][i + 1] + \
                 arr_in[k + 1][j][i - 1] + \
                 arr_in[k - 1][j][i - 1] + \
                 arr_in[k][j + 1][i + 1] + \
                 arr_in[k][j - 1][i + 1] + \
                 arr_in[k][j + 1][i - 1] + \
                 arr_in[k][j - 1][i - 1]) + \
       MPI_C6 * (arr_in[k + 1][j + 1][i + 1] + \
                 arr_in[k - 1][j + 1][i + 1] + \
                 arr_in[k + 1][j - 1][i + 1] + \
                 arr_in[k - 1][j - 1][i + 1] + \
                 arr_in[k + 1][j + 1][i - 1] + \
                 arr_in[k - 1][j + 1][i - 1] + \
                 arr_in[k + 1][j - 1][i - 1] + \
                 arr_in[k - 1][j - 1][i - 1]) )
#define ST_CUBE_BRICK_GPU bOut[b][k][j][i] = ( \
       MPI_C0 * bIn[b][k][j][i] + \
       MPI_C1 * (bIn[b][k + 1][j][i] + \
                 bIn[b][k - 1][j][i] + \
                 bIn[b][k][j + 1][i] + \
                 bIn[b][k][j - 1][i] + \
                 bIn[b][k][j][i + 1] + \
                 bIn[b][k][j][i - 1]) + \
       MPI_C3 * (bIn[b][k + 1][j + 1][i] + \
                 bIn[b][k - 1][j + 1][i] + \
                 bIn[b][k + 1][j - 1][i] + \
                 bIn[b][k - 1][j - 1][i] + \
                 bIn[b][k + 1][j][i + 1] + \
                 bIn[b][k - 1][j][i + 1] + \
                 bIn[b][k + 1][j][i - 1] + \
                 bIn[b][k - 1][j][i - 1] + \
                 bIn[b][k][j + 1][i + 1] + \
                 bIn[b][k][j - 1][i + 1] + \
                 bIn[b][k][j + 1][i - 1] + \
                 bIn[b][k][j - 1][i - 1]) + \
       MPI_C6 * (bIn[b][k + 1][j + 1][i + 1] + \
                 bIn[b][k - 1][j + 1][i + 1] + \
                 bIn[b][k + 1][j - 1][i + 1] + \
                 bIn[b][k - 1][j - 1][i + 1] + \
                 bIn[b][k + 1][j + 1][i - 1] + \
                 bIn[b][k - 1][j + 1][i - 1] + \
                 bIn[b][k + 1][j - 1][i - 1] + \
                 bIn[b][k - 1][j - 1][i - 1]) )
#elif defined(CUBE_125PT)
#define ST_CUBE_CPU arr_out[k][j][i] = ( \
       MPI_C0 * arr_in[k][j][i] + \
       MPI_C1 * (arr_in[k + 1][j][i] + \
                 arr_in[k - 1][j][i] + \
                 arr_in[k][j + 1][i] + \
                 arr_in[k][j - 1][i] + \
                 arr_in[k][j][i + 1] + \
                 arr_in[k][j][i - 1]) + \
       MPI_C2 * (arr_in[k + 2][j][i] + \
                 arr_in[k - 2][j][i] + \
                 arr_in[k][j + 2][i] + \
                 arr_in[k][j - 2][i] + \
                 arr_in[k][j][i + 2] + \
                 arr_in[k][j][i - 2]) + \
       MPI_C3 * (arr_in[k + 1][j + 1][i] + \
                 arr_in[k - 1][j + 1][i] + \
                 arr_in[k + 1][j - 1][i] + \
                 arr_in[k - 1][j - 1][i] + \
                 arr_in[k + 1][j][i + 1] + \
                 arr_in[k - 1][j][i + 1] + \
                 arr_in[k + 1][j][i - 1] + \
                 arr_in[k - 1][j][i - 1] + \
                 arr_in[k][j + 1][i + 1] + \
                 arr_in[k][j - 1][i + 1] + \
                 arr_in[k][j + 1][i - 1] + \
                 arr_in[k][j - 1][i - 1]) + \
       MPI_C4 * (arr_in[k + 1][j + 2][i] + \
                 arr_in[k - 1][j + 2][i] + \
                 arr_in[k + 1][j - 2][i] + \
                 arr_in[k - 1][j - 2][i] + \
                 arr_in[k + 1][j][i + 2] + \
                 arr_in[k - 1][j][i + 2] + \
                 arr_in[k + 1][j][i - 2] + \
                 arr_in[k - 1][j][i - 2] + \
                 arr_in[k][j + 1][i + 2] + \
                 arr_in[k][j - 1][i + 2] + \
                 arr_in[k][j + 1][i - 2] + \
                 arr_in[k][j - 1][i - 2] + \
                 arr_in[k + 2][j + 1][i] + \
                 arr_in[k - 2][j + 1][i] + \
                 arr_in[k + 2][j - 1][i] + \
                 arr_in[k - 2][j - 1][i] + \
                 arr_in[k + 2][j][i + 1] + \
                 arr_in[k - 2][j][i + 1] + \
                 arr_in[k + 2][j][i - 1] + \
                 arr_in[k - 2][j][i - 1] + \
                 arr_in[k][j + 2][i + 1] + \
                 arr_in[k][j - 2][i + 1] + \
                 arr_in[k][j + 2][i - 1] + \
                 arr_in[k][j - 2][i - 1]) + \
       MPI_C5 * (arr_in[k + 2][j + 2][i] + \
                 arr_in[k - 2][j + 2][i] + \
                 arr_in[k + 2][j - 2][i] + \
                 arr_in[k - 2][j - 2][i] + \
                 arr_in[k + 2][j][i + 2] + \
                 arr_in[k - 2][j][i + 2] + \
                 arr_in[k + 2][j][i - 2] + \
                 arr_in[k - 2][j][i - 2] + \
                 arr_in[k][j + 2][i + 2] + \
                 arr_in[k][j - 2][i + 2] + \
                 arr_in[k][j + 2][i - 2] + \
                 arr_in[k][j - 2][i - 2]) + \
       MPI_C6 * (arr_in[k + 1][j + 1][i + 1] + \
                 arr_in[k - 1][j + 1][i + 1] + \
                 arr_in[k + 1][j - 1][i + 1] + \
                 arr_in[k - 1][j - 1][i + 1] + \
                 arr_in[k + 1][j + 1][i - 1] + \
                 arr_in[k - 1][j + 1][i - 1] + \
                 arr_in[k + 1][j - 1][i - 1] + \
                 arr_in[k - 1][j - 1][i - 1]) + \
       MPI_C7 * (arr_in[k + 1][j + 1][i + 2] + \
                 arr_in[k - 1][j + 1][i + 2] + \
                 arr_in[k + 1][j - 1][i + 2] + \
                 arr_in[k - 1][j - 1][i + 2] + \
                 arr_in[k + 1][j + 1][i - 2] + \
                 arr_in[k - 1][j + 1][i - 2] + \
                 arr_in[k + 1][j - 1][i - 2] + \
                 arr_in[k - 1][j - 1][i - 2] + \
                 arr_in[k + 1][j + 2][i + 1] + \
                 arr_in[k - 1][j + 2][i + 1] + \
                 arr_in[k + 1][j - 2][i + 1] + \
                 arr_in[k - 1][j - 2][i + 1] + \
                 arr_in[k + 1][j + 2][i - 1] + \
                 arr_in[k - 1][j + 2][i - 1] + \
                 arr_in[k + 1][j - 2][i - 1] + \
                 arr_in[k - 1][j - 2][i - 1] + \
                 arr_in[k + 2][j + 1][i + 1] + \
                 arr_in[k - 2][j + 1][i + 1] + \
                 arr_in[k + 2][j - 1][i + 1] + \
                 arr_in[k - 2][j - 1][i + 1] + \
                 arr_in[k + 2][j + 1][i - 1] + \
                 arr_in[k - 2][j + 1][i - 1] + \
                 arr_in[k + 2][j - 1][i - 1] + \
                 arr_in[k - 2][j - 1][i - 1]) + \
       MPI_C8 * (arr_in[k + 2][j + 2][i + 1] + \
                 arr_in[k - 2][j + 2][i + 1] + \
                 arr_in[k + 2][j - 2][i + 1] + \
                 arr_in[k - 2][j - 2][i + 1] + \
                 arr_in[k + 2][j + 2][i - 1] + \
                 arr_in[k - 2][j + 2][i - 1] + \
                 arr_in[k + 2][j - 2][i - 1] + \
                 arr_in[k - 2][j - 2][i - 1] + \
                 arr_in[k + 2][j + 1][i + 2] + \
                 arr_in[k - 2][j + 1][i + 2] + \
                 arr_in[k + 2][j - 1][i + 2] + \
                 arr_in[k - 2][j - 1][i + 2] + \
                 arr_in[k + 2][j + 1][i - 2] + \
                 arr_in[k - 2][j + 1][i - 2] + \
                 arr_in[k + 2][j - 1][i - 2] + \
                 arr_in[k - 2][j - 1][i - 2] + \
                 arr_in[k + 1][j + 2][i + 2] + \
                 arr_in[k - 1][j + 2][i + 2] + \
                 arr_in[k + 1][j - 2][i + 2] + \
                 arr_in[k - 1][j - 2][i + 2] + \
                 arr_in[k + 1][j + 2][i - 2] + \
                 arr_in[k - 1][j + 2][i - 2] + \
                 arr_in[k + 1][j - 2][i - 2] + \
                 arr_in[k - 1][j - 2][i - 2]) + \
       MPI_C9 * (arr_in[k + 2][j + 2][i + 2] + \
                 arr_in[k - 2][j + 2][i + 2] + \
                 arr_in[k + 2][j - 2][i + 2] + \
                 arr_in[k - 2][j - 2][i + 2] + \
                 arr_in[k + 2][j + 2][i - 2] + \
                 arr_in[k - 2][j + 2][i - 2] + \
                 arr_in[k + 2][j - 2][i - 2] + \
                 arr_in[k - 2][j - 2][i - 2]) )
#define ST_CUBE_ARR_GPU arr_out[k][j][i] = ( \
       MPI_C0 * arr_in[k][j][i] + \
       MPI_C1 * (arr_in[k + 1][j][i] + \
                 arr_in[k - 1][j][i] + \
                 arr_in[k][j + 1][i] + \
                 arr_in[k][j - 1][i] + \
                 arr_in[k][j][i + 1] + \
                 arr_in[k][j][i - 1]) + \
       MPI_C2 * (arr_in[k + 2][j][i] + \
                 arr_in[k - 2][j][i] + \
                 arr_in[k][j + 2][i] + \
                 arr_in[k][j - 2][i] + \
                 arr_in[k][j][i + 2] + \
                 arr_in[k][j][i - 2]) + \
       MPI_C3 * (arr_in[k + 1][j + 1][i] + \
                 arr_in[k - 1][j + 1][i] + \
                 arr_in[k + 1][j - 1][i] + \
                 arr_in[k - 1][j - 1][i] + \
                 arr_in[k + 1][j][i + 1] + \
                 arr_in[k - 1][j][i + 1] + \
                 arr_in[k + 1][j][i - 1] + \
                 arr_in[k - 1][j][i - 1] + \
                 arr_in[k][j + 1][i + 1] + \
                 arr_in[k][j - 1][i + 1] + \
                 arr_in[k][j + 1][i - 1] + \
                 arr_in[k][j - 1][i - 1]) + \
       MPI_C4 * (arr_in[k + 1][j + 2][i] + \
                 arr_in[k - 1][j + 2][i] + \
                 arr_in[k + 1][j - 2][i] + \
                 arr_in[k - 1][j - 2][i] + \
                 arr_in[k + 1][j][i + 2] + \
                 arr_in[k - 1][j][i + 2] + \
                 arr_in[k + 1][j][i - 2] + \
                 arr_in[k - 1][j][i - 2] + \
                 arr_in[k][j + 1][i + 2] + \
                 arr_in[k][j - 1][i + 2] + \
                 arr_in[k][j + 1][i - 2] + \
                 arr_in[k][j - 1][i - 2] + \
                 arr_in[k + 2][j + 1][i] + \
                 arr_in[k - 2][j + 1][i] + \
                 arr_in[k + 2][j - 1][i] + \
                 arr_in[k - 2][j - 1][i] + \
                 arr_in[k + 2][j][i + 1] + \
                 arr_in[k - 2][j][i + 1] + \
                 arr_in[k + 2][j][i - 1] + \
                 arr_in[k - 2][j][i - 1] + \
                 arr_in[k][j + 2][i + 1] + \
                 arr_in[k][j - 2][i + 1] + \
                 arr_in[k][j + 2][i - 1] + \
                 arr_in[k][j - 2][i - 1]) + \
       MPI_C5 * (arr_in[k + 2][j + 2][i] + \
                 arr_in[k - 2][j + 2][i] + \
                 arr_in[k + 2][j - 2][i] + \
                 arr_in[k - 2][j - 2][i] + \
                 arr_in[k + 2][j][i + 2] + \
                 arr_in[k - 2][j][i + 2] + \
                 arr_in[k + 2][j][i - 2] + \
                 arr_in[k - 2][j][i - 2] + \
                 arr_in[k][j + 2][i + 2] + \
                 arr_in[k][j - 2][i + 2] + \
                 arr_in[k][j + 2][i - 2] + \
                 arr_in[k][j - 2][i - 2]) + \
       MPI_C6 * (arr_in[k + 1][j + 1][i + 1] + \
                 arr_in[k - 1][j + 1][i + 1] + \
                 arr_in[k + 1][j - 1][i + 1] + \
                 arr_in[k - 1][j - 1][i + 1] + \
                 arr_in[k + 1][j + 1][i - 1] + \
                 arr_in[k - 1][j + 1][i - 1] + \
                 arr_in[k + 1][j - 1][i - 1] + \
                 arr_in[k - 1][j - 1][i - 1]) + \
       MPI_C7 * (arr_in[k + 1][j + 1][i + 2] + \
                 arr_in[k - 1][j + 1][i + 2] + \
                 arr_in[k + 1][j - 1][i + 2] + \
                 arr_in[k - 1][j - 1][i + 2] + \
                 arr_in[k + 1][j + 1][i - 2] + \
                 arr_in[k - 1][j + 1][i - 2] + \
                 arr_in[k + 1][j - 1][i - 2] + \
                 arr_in[k - 1][j - 1][i - 2] + \
                 arr_in[k + 1][j + 2][i + 1] + \
                 arr_in[k - 1][j + 2][i + 1] + \
                 arr_in[k + 1][j - 2][i + 1] + \
                 arr_in[k - 1][j - 2][i + 1] + \
                 arr_in[k + 1][j + 2][i - 1] + \
                 arr_in[k - 1][j + 2][i - 1] + \
                 arr_in[k + 1][j - 2][i - 1] + \
                 arr_in[k - 1][j - 2][i - 1] + \
                 arr_in[k + 2][j + 1][i + 1] + \
                 arr_in[k - 2][j + 1][i + 1] + \
                 arr_in[k + 2][j - 1][i + 1] + \
                 arr_in[k - 2][j - 1][i + 1] + \
                 arr_in[k + 2][j + 1][i - 1] + \
                 arr_in[k - 2][j + 1][i - 1] + \
                 arr_in[k + 2][j - 1][i - 1] + \
                 arr_in[k - 2][j - 1][i - 1]) + \
       MPI_C8 * (arr_in[k + 2][j + 2][i + 1] + \
                 arr_in[k - 2][j + 2][i + 1] + \
                 arr_in[k + 2][j - 2][i + 1] + \
                 arr_in[k - 2][j - 2][i + 1] + \
                 arr_in[k + 2][j + 2][i - 1] + \
                 arr_in[k - 2][j + 2][i - 1] + \
                 arr_in[k + 2][j - 2][i - 1] + \
                 arr_in[k - 2][j - 2][i - 1] + \
                 arr_in[k + 2][j + 1][i + 2] + \
                 arr_in[k - 2][j + 1][i + 2] + \
                 arr_in[k + 2][j - 1][i + 2] + \
                 arr_in[k - 2][j - 1][i + 2] + \
                 arr_in[k + 2][j + 1][i - 2] + \
                 arr_in[k - 2][j + 1][i - 2] + \
                 arr_in[k + 2][j - 1][i - 2] + \
                 arr_in[k - 2][j - 1][i - 2] + \
                 arr_in[k + 1][j + 2][i + 2] + \
                 arr_in[k - 1][j + 2][i + 2] + \
                 arr_in[k + 1][j - 2][i + 2] + \
                 arr_in[k - 1][j - 2][i + 2] + \
                 arr_in[k + 1][j + 2][i - 2] + \
                 arr_in[k - 1][j + 2][i - 2] + \
                 arr_in[k + 1][j - 2][i - 2] + \
                 arr_in[k - 1][j - 2][i - 2]) + \
       MPI_C9 * (arr_in[k + 2][j + 2][i + 2] + \
                 arr_in[k - 2][j + 2][i + 2] + \
                 arr_in[k + 2][j - 2][i + 2] + \
                 arr_in[k - 2][j - 2][i + 2] + \
                 arr_in[k + 2][j + 2][i - 2] + \
                 arr_in[k - 2][j + 2][i - 2] + \
                 arr_in[k + 2][j - 2][i - 2] + \
                 arr_in[k - 2][j - 2][i - 2]) )
#define ST_CUBE_BRICK_GPU bOut[b][k][j][i] = ( \
       MPI_C0 * bIn[b][k][j][i] + \
       MPI_C1 * (bIn[b][k + 1][j][i] + \
                 bIn[b][k - 1][j][i] + \
                 bIn[b][k][j + 1][i] + \
                 bIn[b][k][j - 1][i] + \
                 bIn[b][k][j][i + 1] + \
                 bIn[b][k][j][i - 1]) + \
       MPI_C2 * (bIn[b][k + 2][j][i] + \
                 bIn[b][k - 2][j][i] + \
                 bIn[b][k][j + 2][i] + \
                 bIn[b][k][j - 2][i] + \
                 bIn[b][k][j][i + 2] + \
                 bIn[b][k][j][i - 2]) + \
       MPI_C3 * (bIn[b][k + 1][j + 1][i] + \
                 bIn[b][k - 1][j + 1][i] + \
                 bIn[b][k + 1][j - 1][i] + \
                 bIn[b][k - 1][j - 1][i] + \
                 bIn[b][k + 1][j][i + 1] + \
                 bIn[b][k - 1][j][i + 1] + \
                 bIn[b][k + 1][j][i - 1] + \
                 bIn[b][k - 1][j][i - 1] + \
                 bIn[b][k][j + 1][i + 1] + \
                 bIn[b][k][j - 1][i + 1] + \
                 bIn[b][k][j + 1][i - 1] + \
                 bIn[b][k][j - 1][i - 1]) + \
       MPI_C4 * (bIn[b][k + 1][j + 2][i] + \
                 bIn[b][k - 1][j + 2][i] + \
                 bIn[b][k + 1][j - 2][i] + \
                 bIn[b][k - 1][j - 2][i] + \
                 bIn[b][k + 1][j][i + 2] + \
                 bIn[b][k - 1][j][i + 2] + \
                 bIn[b][k + 1][j][i - 2] + \
                 bIn[b][k - 1][j][i - 2] + \
                 bIn[b][k][j + 1][i + 2] + \
                 bIn[b][k][j - 1][i + 2] + \
                 bIn[b][k][j + 1][i - 2] + \
                 bIn[b][k][j - 1][i - 2] + \
                 bIn[b][k + 2][j + 1][i] + \
                 bIn[b][k - 2][j + 1][i] + \
                 bIn[b][k + 2][j - 1][i] + \
                 bIn[b][k - 2][j - 1][i] + \
                 bIn[b][k + 2][j][i + 1] + \
                 bIn[b][k - 2][j][i + 1] + \
                 bIn[b][k + 2][j][i - 1] + \
                 bIn[b][k - 2][j][i - 1] + \
                 bIn[b][k][j + 2][i + 1] + \
                 bIn[b][k][j - 2][i + 1] + \
                 bIn[b][k][j + 2][i - 1] + \
                 bIn[b][k][j - 2][i - 1]) + \
       MPI_C5 * (bIn[b][k + 2][j + 2][i] + \
                 bIn[b][k - 2][j + 2][i] + \
                 bIn[b][k + 2][j - 2][i] + \
                 bIn[b][k - 2][j - 2][i] + \
                 bIn[b][k + 2][j][i + 2] + \
                 bIn[b][k - 2][j][i + 2] + \
                 bIn[b][k + 2][j][i - 2] + \
                 bIn[b][k - 2][j][i - 2] + \
                 bIn[b][k][j + 2][i + 2] + \
                 bIn[b][k][j - 2][i + 2] + \
                 bIn[b][k][j + 2][i - 2] + \
                 bIn[b][k][j - 2][i - 2]) + \
       MPI_C6 * (bIn[b][k + 1][j + 1][i + 1] + \
                 bIn[b][k - 1][j + 1][i + 1] + \
                 bIn[b][k + 1][j - 1][i + 1] + \
                 bIn[b][k - 1][j - 1][i + 1] + \
                 bIn[b][k + 1][j + 1][i - 1] + \
                 bIn[b][k - 1][j + 1][i - 1] + \
                 bIn[b][k + 1][j - 1][i - 1] + \
                 bIn[b][k - 1][j - 1][i - 1]) + \
       MPI_C7 * (bIn[b][k + 1][j + 1][i + 2] + \
                 bIn[b][k - 1][j + 1][i + 2] + \
                 bIn[b][k + 1][j - 1][i + 2] + \
                 bIn[b][k - 1][j - 1][i + 2] + \
                 bIn[b][k + 1][j + 1][i - 2] + \
                 bIn[b][k - 1][j + 1][i - 2] + \
                 bIn[b][k + 1][j - 1][i - 2] + \
                 bIn[b][k - 1][j - 1][i - 2] + \
                 bIn[b][k + 1][j + 2][i + 1] + \
                 bIn[b][k - 1][j + 2][i + 1] + \
                 bIn[b][k + 1][j - 2][i + 1] + \
                 bIn[b][k - 1][j - 2][i + 1] + \
                 bIn[b][k + 1][j + 2][i - 1] + \
                 bIn[b][k - 1][j + 2][i - 1] + \
                 bIn[b][k + 1][j - 2][i - 1] + \
                 bIn[b][k - 1][j - 2][i - 1] + \
                 bIn[b][k + 2][j + 1][i + 1] + \
                 bIn[b][k - 2][j + 1][i + 1] + \
                 bIn[b][k + 2][j - 1][i + 1] + \
                 bIn[b][k - 2][j - 1][i + 1] + \
                 bIn[b][k + 2][j + 1][i - 1] + \
                 bIn[b][k - 2][j + 1][i - 1] + \
                 bIn[b][k + 2][j - 1][i - 1] + \
                 bIn[b][k - 2][j - 1][i - 1]) + \
       MPI_C8 * (bIn[b][k + 2][j + 2][i + 1] + \
                 bIn[b][k - 2][j + 2][i + 1] + \
                 bIn[b][k + 2][j - 2][i + 1] + \
                 bIn[b][k - 2][j - 2][i + 1] + \
                 bIn[b][k + 2][j + 2][i - 1] + \
                 bIn[b][k - 2][j + 2][i - 1] + \
                 bIn[b][k + 2][j - 2][i - 1] + \
                 bIn[b][k - 2][j - 2][i - 1] + \
                 bIn[b][k + 2][j + 1][i + 2] + \
                 bIn[b][k - 2][j + 1][i + 2] + \
                 bIn[b][k + 2][j - 1][i + 2] + \
                 bIn[b][k - 2][j - 1][i + 2] + \
                 bIn[b][k + 2][j + 1][i - 2] + \
                 bIn[b][k - 2][j + 1][i - 2] + \
                 bIn[b][k + 2][j - 1][i - 2] + \
                 bIn[b][k - 2][j - 1][i - 2] + \
                 bIn[b][k + 1][j + 2][i + 2] + \
                 bIn[b][k - 1][j + 2][i + 2] + \
                 bIn[b][k + 1][j - 2][i + 2] + \
                 bIn[b][k - 1][j - 2][i + 2] + \
                 bIn[b][k + 1][j + 2][i - 2] + \
                 bIn[b][k - 1][j + 2][i - 2] + \
                 bIn[b][k + 1][j - 2][i - 2] + \
                 bIn[b][k - 1][j - 2][i - 2]) + \
       MPI_C9 * (bIn[b][k + 2][j + 2][i + 2] + \
                 bIn[b][k - 2][j + 2][i + 2] + \
                 bIn[b][k + 2][j - 2][i + 2] + \
                 bIn[b][k - 2][j - 2][i + 2] + \
                 bIn[b][k + 2][j + 2][i - 2] + \
                 bIn[b][k - 2][j + 2][i - 2] + \
                 bIn[b][k + 2][j - 2][i - 2] + \
                 bIn[b][k - 2][j - 2][i - 2]) )
#else
#endif


#endif //BRICK_MACROS_COEFFS_H
