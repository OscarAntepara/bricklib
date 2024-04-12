#ifndef MG_COMMON_H
#define MG_COMMON_H

#include <vector>
#include <brick.h>

/**
 * @brief Create an multidimensional array initialized with a constant
 * @param[in] list dimensions
 * @return pointer to the newly created array
 */
static bElem *constantArray(const std::vector<long> &list, bElem const_val) {
  long size;
  bElem *arr = uninitArray(list, size);
#pragma omp parallel for
  for (long l = 0; l < size; ++l)
    arr[l] = const_val;

  return arr;
}

/**
 * @brief Create an multidimensional array initialized with a product of sin function
 * @param[in] list dimensions
 * @return pointer to the newly created array
 */
bElem *prodSinArray(const bElem &coef, const std::vector<long> &list, int padding, int gz) {
  long size;
  bElem *arr = uninitArray(list, size);
//#pragma omp parallel for
  for (long l = 0; l < size; ++l)
  {
    arr[l] = 0.0;
  }
  auto arr_in = (bElem (*)[list[1]][list[0]]) arr;
  bElem inner_dom_size = list[0] - (2*(padding + gz));
  bElem pi = 3.14159265358979323846;
  for (long k = -gz-padding; k < list[2]-((1*padding) + gz); ++k)
    for (long j = -gz-padding; j < list[1]-((1*padding) + gz); ++j)
      for (long i = -gz-padding; i < list[0]-((1*padding) + gz); ++i) {
        bElem h = 1.0/inner_dom_size;
        bElem x = h*(.5+i);
        bElem y = h*(.5+j);
        bElem z = h*(.5+k);
        bElem value = coef*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z);
	      arr_in[padding+gz+k][padding+gz+j][padding+gz+i] = value;
      }

  return arr;
}

/**
 * @brief Create an multidimensional array initialized with a polynomial
 * @param[in] list dimensions
 * @return pointer to the newly created array
 */
static bElem *polyArray(const std::vector<long> &list, int padding, int gz, int ilevel) {
  long size;
  bElem *arr = uninitArray(list, size);
#pragma omp parallel for
  for (long l = 0; l < size; ++l)
  {
    arr[l] = 0.0;
  }
  auto arr_in = (bElem (*)[list[1]][list[0]]) arr;
  bElem inner_dom_size = list[0] - (2*(padding + gz));
  bElem one_brick_lenght = 8.0*(1.0/inner_dom_size);
  if (ilevel<=1) one_brick_lenght = 0.;
  bElem low_coord = 0.3125;
  bElem high_coord = 0.6875;
  for (long k = -gz-padding; k < list[2]-((1*padding) + gz); ++k)
    for (long j = -gz-padding; j < list[1]-((1*padding) + gz); ++j)
      for (long i = -gz-padding; i < list[0]-((1*padding) + gz); ++i) {
        bElem h = 1.0/inner_dom_size;
        bElem x = h*(.5+i);
        bElem y = h*(.5+j);
        bElem z = h*(.5+k);
        //bElem value = ((x-0.5)*(x-0.5)) + ((y-0.5)*(y-0.5)) + ((z-0.5)*(z-0.5));
        bElem value = 0.0;
        //if (((x-0.5)*(x-0.5)) + ((y-0.5)*(y-0.5)) + ((z-0.5)*(z-0.5))<0.25*0.25) value =1.0;
        if (x>low_coord-one_brick_lenght && x<high_coord+one_brick_lenght 
            && y>low_coord-one_brick_lenght && y<high_coord+one_brick_lenght 
            && z>low_coord-one_brick_lenght && z<high_coord+one_brick_lenght) value =1.0;
        //if (y<0.5) value =1.0;
        //value = abs(value - 1);
        //value = 1.0;
	      arr_in[padding+gz+k][padding+gz+j][padding+gz+i] = value;
      }

  return arr;
}

/**
 * @brief Create an multidimensional array initialized with a red-black organization
 * @param[in] list dimensions
 * @return pointer to the newly created array
 */
static bElem *redBlackArray(const std::vector<long> &list, int padding, int gz) {
  long size;
  bElem *arr = uninitArray(list, size);
#pragma omp parallel for
  for (long l = 0; l < size; ++l)
  {
    arr[l] = 0.0;
  }
  auto arr_in = (bElem (*)[list[1]][list[0]]) arr;
#pragma omp parallel for
  for (long k = -gz-padding; k < list[2]-((1*padding) + gz); ++k)
    for (long j = -gz-padding; j < list[1]-((1*padding) + gz); ++j)
      for (long i = -gz-padding; i < list[0]-((1*padding) + gz); ++i) {
        bElem value = 1.0;
        if((i+padding+gz^j+padding+gz^k+padding+gz^1)&0x1){
          value=1.0;
        }else{
          value=0.0;
        }
        //std::cout<<"i "<<padding+gz+i<<" j "<<padding+gz+j<<" k "<<padding+gz+k<<" val "<<value<<std::endl;
	      arr_in[padding+gz+k][padding+gz+j][padding+gz+i] = value;
      }

  return arr;
}


/**
 * @brief Create an multidimensional array initialized with a red-black organization for bricks
 * @param[in] list dimensions
 * @return pointer to the newly created array
 */
static unsigned *redBlackBricks(const std::vector<long> &list, int padding, int gz, unsigned* grid, int color) {
  long size = 1;
  for (auto i: list)
    size *= i;
  size = (size + 1)/2;
  unsigned *arr = (unsigned *) aligned_alloc(ALIGN, size * sizeof(unsigned));
#pragma omp parallel for
  for (long l = 0; l < size; ++l)
  {
    arr[l] = 0.0;
  }
  size = 0;
#pragma omp parallel for
  for (long k = -gz-padding; k < list[2]-((1*padding) + gz); ++k)
    for (long j = -gz-padding; j < list[1]-((1*padding) + gz); ++j)
      for (long i = -gz-padding; i < list[0]-((1*padding) + gz); ++i) {
        unsigned value = 0;
        unsigned b = grid[i+padding+gz + (j+padding+gz + (k+padding+gz) * list[1]) * list[0]];
        if((i+padding+gz^j+padding+gz^k+padding+gz^1)&0x1){
          value=1;
        }
        if(value==1 && color==1){
	        arr[size] = b;
          size++;
        }else if(value==0 && color==0){
	        arr[size] = b;
          size++;
        }
      }

  return arr;
}


//------------------------------------------------------------------------------------------------------------------------------
#endif