if (HAS_SYCL)
    set(SYCL3AXIS_INCLUDE_DIRS
            "${SYCL_DIR}/include"
            "${SYCL_DIR}/include/sycl")
    if (${CODEGEN_ARCH} MATCHES "INTEL")
        set(SYCL3AXIS_COMPILE_OPTIONS
            "-fsycl")
    elseif (${CODEGEN_ARCH} MATCHES "NVIDIA")
        set(SYCL3AXIS_COMPILE_OPTIONS
            "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend '--cuda-gpu-arch=sm_80'")
    elseif (${CODEGEN_ARCH} MATCHES "AMD")
        set(SYCL3AXIS_COMPILE_OPTIONS
            "-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a")
    endif ()
    VSTARGET(SYCL3AXIS d3_stencils.sycl.cpp d3_stencils-out.sycl.cpp)
    add_executable(single-perf-sycl sycl.cpp ${VSTARGET_SYCL3AXIS_OUTPUT})
    target_link_libraries(single-perf-sycl brick brickhelper stdc++ sycl)
    target_include_directories(single-perf-sycl
            PUBLIC "${SYCL_DIR}/include"
            PUBLIC "${SYCL_DIR}/include/sycl"
            PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/stencils")
    target_link_directories(single-perf-sycl PUBLIC "${SYCL_DIR}/lib")
    # sycl need option for both phase
    if (${CODEGEN_ARCH} MATCHES "INTEL")
        target_compile_options(single-perf-sycl PUBLIC -fsycl)
        target_link_options(single-perf-sycl PUBLIC -fsycl)
    elseif (${CODEGEN_ARCH} MATCHES "NVIDIA")
        target_compile_options(single-perf-sycl PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend '--cuda-gpu-arch=sm_80')
        target_link_options(single-perf-sycl PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend '--cuda-gpu-arch=sm_80')
    elseif (${CODEGEN_ARCH} MATCHES "AMD")
        target_compile_options(single-perf-sycl PUBLIC -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a)
        target_link_options(single-perf-sycl PUBLIC -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a)
    endif ()
    set_target_properties(single-perf-sycl
            PROPERTIES OUTPUT_NAME sycl)
endif ()

if (CMAKE_CUDA_COMPILER)
    set(C3AXIS_INCLUDE_DIRS
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
    VSTARGET(C3AXIS d3_stencils.cu d3_stencils-out.cu)
    add_executable(single-perf-cuda cuda.cpp ${VSTARGET_C3AXIS_OUTPUT})
    target_link_libraries(single-perf-cuda brickhelper)
    set_target_properties(single-perf-cuda
            PROPERTIES OUTPUT_NAME cuda)
    target_include_directories(single-perf-cuda PUBLIC
            "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            )
endif ()

if (BRICK_USE_HIP)
    set(H3AXIS_INCLUDE_DIRS "${HIP_PATH}/include")
    VSTARGET(H3AXIS d3_stencils.hip.cpp d3_stencils-out.hip.cpp)
    set(HIP_SRCS hip.cpp ${VSTARGET_H3AXIS_OUTPUT})
    set_source_files_properties(${HIP_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(single-perf-hip ${HIP_SRCS} CLANG_OPTIONS ${CMAKE_CXX_FLAGS})
    target_link_libraries(single-perf-hip brickhelper hip::host)
    set_target_properties(single-perf-hip
            PROPERTIES OUTPUT_NAME hip)
endif ()

if (USE_DPC)
   VSTARGET(D3AXIS ../stencils/3axis_dpc.cpp ../stencils/3axis_dpc-out.cpp)
   set(CMAKE_CXX_COMPILER "dpcpp")
   set(CMAKE_LINKER "dpcpp") 
   set(CMAKE_AR "llvm-ar") 
   set(CMAKE_RANLIB "llvm-ranlib") 
   set(CMAKE_CXX_STANDARD 17) 
   add_executable(single-dpc dpc.cpp ${VSTARGET_D3AXIS_OUTPUT})
   target_link_libraries(single-dpc brickhelper)
   target_include_directories(single-dpc PUBLIC "$(which dpcpp)/../../include/sycl/")
   set_target_properties(single-dpc PROPERTIES OUTPUT_NAME dpc)
endif ()

