#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <cuda_runtime.h>
#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        // Initialize MPI
        MPI_Init(&argc, &argv);
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // Set CUDA device
        cudaError_t err = cudaSetDevice(rank);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error setting device " << rank << ": " << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        // Create TCPStore options
        c10d::TCPStoreOptions opts;
        opts.isServer = (rank == 0);
        opts.numWorkers = world_size;
        opts.port = 29500;

        // Create store
        auto store = c10::make_intrusive<c10d::TCPStore>("localhost", opts);

        // Create options
        auto options = c10d::ProcessGroupNCCL::Options::create();

        // Initialize process group
        auto pg = c10::make_intrusive<c10d::ProcessGroupNCCL>(store, rank, world_size, options);

        // Create tensor
        torch::Tensor tensor = torch::ones({5}, torch::device(torch::kCUDA)) * (rank + 1);

        // Perform all-reduce
        std::vector<torch::Tensor> tensors = {tensor};
        auto work = pg->allreduce(tensors);
        work->wait();

        // Print result
        if (rank == 0) {
            std::cout << "All-reduce result: " << tensors[0] << std::endl;
        }

        // Cleanup
        MPI_Finalize();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}