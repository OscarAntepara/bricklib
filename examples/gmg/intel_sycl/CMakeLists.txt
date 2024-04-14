if (HAS_SYCL)
    set(SYCL3AXIS_INCLUDE_DIRS
            "${SYCL_DIR}/include"
            "${SYCL_DIR}/include/sycl")
    set(SYCL3AXIS_COMPILE_OPTIONS
        "-fsycl")
    VSTARGET(SYCL3AXIS mg_brick.sycl.cpp mg_brick-out.sycl.cpp)
    set(SYCL_SRCS main.sycl.cpp ${VSTARGET_SYCL3AXIS_OUTPUT} args.cpp)
    add_executable(gmg-sycl ${SYCL_SRCS} ${VSTARGET_SYCL3AXIS_OUTPUT})
    target_link_libraries(gmg-sycl brick brick-mpi brickhelper stdc++ sycl ${MPI_CXX_LIBRARIES} ${LIBRT})
    target_include_directories(gmg-sycl
            PUBLIC "${SYCL_DIR}/include"
            PUBLIC "${SYCL_DIR}/include/sycl"
            PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/stencils")
    target_link_directories(gmg-sycl PUBLIC "${SYCL_DIR}/lib")
    # sycl need option for both phase
    target_compile_options(gmg-sycl PUBLIC -fsycl)
    target_link_options(gmg-sycl PUBLIC -fsycl)
    set_target_properties(gmg-sycl
            PROPERTIES OUTPUT_NAME sycl)
endif ()

