/**
 * @file
 * @brief Header necessary for SYCL program to include
 */

#ifndef BRICK_BRICK_SYCL_H
#define BRICK_BRICK_SYCL_H

#include "vecscatter.h"
#include "dev_shl.h"
#include <CL/sycl.hpp>
#include "brick.h"

typedef struct oclbrick {
  bElem *dat;
  unsigned *adj;
  unsigned step;
} oclbrick;

#ifndef NDEBUG
static sycl::property_list properties{sycl::property::queue::enable_profiling()};
static sycl::queue gpu_queue = sycl::queue(sycl::gpu_selector(), properties);
static sycl::queue cpu_queue = sycl::queue(sycl::host_selector(), properties);
#else
static sycl::queue gpu_queue = sycl::queue(sycl::gpu_selector());
static sycl::queue cpu_queue = sycl::queue(sycl::host_selector());
#endif

enum syclError_t {
    sycl_success,
    memcpy_failed,
    malloc_failed
};

enum syclMemcpyKind {
    syclMemcpyHostToDevice,
    syclMemcpyDeviceToHost
};

/**
 * @brief Interface to allocate memory on a sycl GPU with a similar footprint to CUDA. 
 * @tparam T Type of data to be store in the buffer, usually implicit.
 * @param buffer Where to store the pointer to the allocated memory.
 * @param size Size of allocated memory in bytes.
 * @return sycl_success (alias of gpuSuccess) if memory was successfully allocated.
 */
template <typename T>
inline syclError_t sycl_malloc(T **buffer, size_t size) {
    T *ptr = (T *) sycl::malloc_device(size, gpu_queue);
    if (ptr == nullptr) {
        return malloc_failed;
    }
    gpu_queue.memset(ptr, 0, size).wait_and_throw();
    (*buffer) = ptr;
    return sycl_success;
}

/**
 * @brief Copy data from host to sycl GPU. If data must be returned to the host after kernel execution, a buffer/accessor pattern is preffered.
 * @tparam T The type of data being copied, usually implicit.
 * @param dst Pointer to the destination on the GPU.
 * @param ptr Pointer to the data source on the host.
 * @param size The size of the data copy in bytes.
 * @param type Currently may only be syclMemcpyHostToDevice (alias of gpuMemcpyHostToDevice)
 * @return sycl_success (alias of gpuSuccess) if memory was successfully allocated.
 */
template <typename T>
inline syclError_t sycl_memcpy(T *dst, T *ptr, size_t size, syclMemcpyKind type) {    
    assert(type == syclMemcpyHostToDevice);
    gpu_queue.memcpy((void *) dst, (void *) ptr, size).wait_and_throw();
    return sycl_success;
}

/**
 * @brief Free allocated memory on a sycl GPU.
 * @tparam T The type of the data being freed, usually implicit.
 * @param ptr Pointer to the previously allocated memory.
 * @return sycl_success (alias of gpuSuccess) if memory was successfully allocated.
 */
template <typename T>
inline syclError_t sycl_free(T *ptr) {
    sycl::free((void *) ptr, gpu_queue);
    return sycl_success;
} 

static const char *sycl_get_error_string(syclError_t e) {
    switch(e) {
        case sycl_success: return "sycl Success";
        case malloc_failed: return "Failed to allocate memory";
        case memcpy_failed: return "Failed to perform memcpy";
        default: return "Unknown error";
    }
}

#define gpuMalloc(p, s) sycl_malloc(p, s)
#define gpuMemcpy(d, p, s, k) sycl_memcpy(d, p, s, k)
#define gpuFree(p) sycl_free(p)
#define gpuMemcpyKind syclMemcpyKind
#define gpuMemcpyHostToDevice syclMemcpyHostToDevice
#define gpuMemcpyDeviceToHost syclMemcpyDeviceToHost
#define gpuSuccess sycl_success
#define gpuGetErrorString(e) sycl_get_error_string(e)

/// Generic base template, see <a href="structBrick_3_01Dim_3_01BDims_8_8_8_01_4_00_01Dim_3_01Folds_8_8_8_01_4_01_4.html">Brick< Dim< BDims... >, Dim< Folds... > ></a>
template<typename...>
struct syclBrick;

/**
 * @brief Brick data structure
 * @tparam BDims The brick dimensions
 * @tparam Folds The fold dimensions
 *
 * Some example usage:
 * @code{.cpp}
 * Brick<Dim<8,8,8>, Dim<2,4>> bIn(&bInfo, bStorage, 0); // 8x8x8 bricks with 2x4 folding
 * bIn[1][0][0][0] = 2; // Setting the first element for the brick at index 1 to 2
 * @endcode
 */
template<
    unsigned ... BDims,
    unsigned ... Folds>
struct syclBrick<Dim<BDims...>, Dim<Folds...> > {
  typedef syclBrick<Dim<BDims...>, Dim<Folds...> > mytype;    ///< Shorthand for this struct's type
  typedef BrickInfo<sizeof...(BDims)> myBrickInfo;        ///< Shorthand for type of the metadata

  static constexpr unsigned VECLEN = cal_size<Folds...>::value;     ///< Vector length shorthand
  static constexpr unsigned BRICKSIZE = cal_size<BDims...>::value;  ///< Brick size shorthand

  myBrickInfo *bInfo;        ///< Pointer to (possibly shared) metadata
  size_t step;             ///< Spacing between bricks in unit of bElem (BrickStorage)
  bElem *dat;                ///< Offsetted memory (BrickStorage)

  /// Indexing operator returns: @ref _BrickAccessor
  FORCUDA
  inline _BrickAccessor<mytype, Dim<BDims...>, Dim<Folds...>,
      typename std::conditional<sizeof...(BDims) == sizeof...(Folds), bool, void>::type> operator[](unsigned b) {
    return _BrickAccessor<mytype, Dim<BDims...>, Dim<Folds...>,
        typename std::conditional<sizeof...(BDims) == sizeof...(Folds), bool, void>::type>(this, b, 0, 0, 0);
  }

  /// Return the adjacency list of brick *b*
  template<unsigned ... Offsets>
  FORCUDA
  inline bElem *neighbor(unsigned b) {
    unsigned off = cal_offs<sizeof...(BDims), Offsets...>::value;
    return &dat[bInfo->adj[b][off] * step];
  }

  syclBrick(myBrickInfo *bInfo, bElem *bData, size_t bStep, unsigned offset) : bInfo(bInfo) {
    dat = bData + offset;
    step = bStep;
  }
};

//#include "brick-gpu.h"

#endif //BRICK_BRICK_SYCL_H
