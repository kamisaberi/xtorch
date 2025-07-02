#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <mpi.h>
#include <iostream>
#include <memory>
#include "distributed_data_parallel.h"

// Neural Network model
struct Net : torch::nn::Module {
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(784, 256));
        fc2 = register_module("fc2", torch::nn::Linear(256, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.view({-1, 784});
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Verify we have 4 processes (2 GPUs + 2 CPUs)
    if (world_size != 4) {
        if (rank == 0) {
            std::cerr << "This example requires exactly 4 processes\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Set device based on rank
    torch::Device device(torch::kCPU);
    if (rank < 2 && torch::cuda::is_available() && torch::cuda::device_count() >= 2) {
        device = torch::Device(torch::kCUDA, rank);
        torch::cuda::set_device(rank);
    }

    // Create model
    auto model = std::make_shared<Net>();

    // Create dataset with distributed sampler
    auto dataset = torch::data::datasets::MNIST("./data")
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    auto sampler = torch::data::samplers::DistributedRandomSampler(
        dataset.size().value(), world_size, rank);

    auto dataloader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(64).workers(2),
        &sampler);

    // Create optimizer
    torch::optim::SGD optimizer(
        model->parameters(),
        torch::optim::SGDOptions(0.01).momentum(0.5));

    // Create DistributedDataParallel
    DistributedDataParallel ddp(*model, world_size, rank, device, 64);

    // Train for 10 epochs
    if (rank == 0) {
        std::cout << "Starting training with 2 GPUs and 2 CPUs\n";
    }
    ddp.train(*dataloader, optimizer, 10);

    // Save model (only from rank 0)
    if (rank == 0) {
        torch::save(model, "mnist_ddp_model.pt");
        std::cout << "Model saved\n";
    }

    // Cleanup
    MPI_Finalize();
    return 0;
}