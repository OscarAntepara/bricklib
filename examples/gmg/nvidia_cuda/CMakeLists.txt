if(CMAKE_CUDA_COMPILER)
    include_directories(
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    VSTARGET(C3MPI mg_brick.cu mg_brick-out.cu)
    add_executable(gmg-cuda main.cu ${VSTARGET_C3MPI_OUTPUT} args.cpp)
    target_compile_definitions(gmg-cuda PRIVATE DECOMP_PAGEUNALIGN)
    target_link_libraries(gmg-cuda brick brick-mpi brickhelper ${MPI_CXX_LIBRARIES} ${LIBRT} /opt/cray/pe/mpich/8.1.28/gtl/lib/libmpi_gtl_cuda.so)
    set_target_properties(gmg-cuda
            PROPERTIES OUTPUT_NAME cuda)

endif ()
