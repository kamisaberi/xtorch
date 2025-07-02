cmake_minimum_required(VERSION 3.10)
project(MyDDPApp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER mpicxx) # Use MPI C++ compiler wrapper

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED) # For CUDA paths and libraries

# Find NCCL (adjust paths if necessary, or set NCCL_HOME env var)
find_path(NCCL_INCLUDE_DIR nccl.h HINTS ENV NCCL_HOME PATH_SUFFIXES include)
find_library(NCCL_LIBRARY NAMES nccl HINTS ENV NCCL_HOME PATH_SUFFIXES lib lib64)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL not found. Please set NCCL_HOME or NCCL_INCLUDE_DIR/NCCL_LIBRARY.")
endif()

add_executable(mpi_nccl_ddp_app
        main_mpi_nccl_ddp.cpp
        my_from_scratch_ddp_with_mpi_nccl.cpp
        # your_model.cpp
        # your_dataloader.cpp (if they have .cpp files)
)

target_include_directories(mpi_nccl_ddp_app PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${NCCL_INCLUDE_DIR}
        # Path to your distributed_data_parallel_base.h, xt::Module, etc.
        ${CMAKE_CURRENT_SOURCE_DIR}/include/base # Example
        ${CMAKE_CURRENT_SOURCE_DIR}             # For current dir headers
)

target_link_libraries(mpi_nccl_ddp_app PRIVATE
        ${TORCH_LIBRARIES}
        # MPI libs are handled by mpicxx
        ${CUDA_LIBRARIES}    # Or specific CUDA libs like cudart
        ${NCCL_LIBRARY}
)
set_property(TARGET mpi_nccl_ddp_app PROPERTY CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")