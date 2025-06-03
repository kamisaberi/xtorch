#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
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
        torch::Device device(torch::kCUDA, rank);
        torch::cuda::set_device(rank);

        // Create store
        auto store = c10::make_intrusive<c10d::TCPStore>("localhost", 29500, world_size, rank == 0);

        // Create options
        auto options = c10d::ProcessGroupNCCL::Options::create();

        // Initialize process group
        auto pg = c10::make_intrusive<c10d::ProcessGroupNCCL>(store, rank, world_size, options);

        // Create tensor
        torch::Tensor tensor = torch::ones({5}).to(device) * (rank + 1);

        // Perform all-reduce
        std::vector<torch::Tensor> tensors = {tensor};
        auto work = pg->allreduce(tensors);
        work->wait();

        // Print result
        if (rank == 0) {
            std::cout << "All-reduce result: " << tensors[0] << std::endl;
            // Expected: [10, 10, 10, 10, 10] for 4 GPUs
        }

        // Cleanup
        MPI_Finalize();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}