#pragma once

#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <vector>
#include <memory>
#include <mutex>

class DistributedDataParallel {
public:
    // Constructor
    DistributedDataParallel(
        torch::nn::Module& model,
        int world_size,
        int rank,
        torch::Device device,
        size_t batch_size)
        : base_model_(model),
          world_size_(world_size),
          rank_(rank),
          device_(device),
          batch_size_(batch_size) {
        initialize();
    }

    // Train the model
    template<typename DataLoader>
    void train(DataLoader& dataloader, torch::optim::Optimizer& optimizer, size_t epochs) {
        // Initialize process group
        auto store = c10d::TCPStore("localhost", 29500, world_size_, rank_ == 0);
        auto process_group = std::make_shared<c10d::ProcessGroupMPI>(store, rank_, world_size_);

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            model_->train();
            size_t local_batch_size = batch_size_ / world_size_;

            // Distributed sampler ensures each process gets unique subset of data
            for (auto& batch : dataloader) {
                auto data = batch.data.to(device_);
                auto target = batch.target.to(device_);

                // Adjust for uneven batch sizes
                auto current_batch_size = data.size(0);
                if (current_batch_size != local_batch_size) {
                    data = data.narrow(0, 0, local_batch_size);
                    target = target.narrow(0, 0, local_batch_size);
                }

                // Forward pass
                optimizer.zero_grad();
                auto output = model_->forward(data);
                auto loss = torch::nn::functional::cross_entropy(output, target);

                // Backward pass
                loss.backward();

                // Synchronize gradients across all processes
                synchronize_gradients(process_group);

                // Update parameters
                optimizer.step();
            }

            // Synchronize after each epoch
            if (rank_ == 0) {
                std::cout << "Epoch " << epoch + 1 << " completed\n";
            }
            torch::cuda::synchronize();
        }
    }

private:
    void initialize() {
        // Move model to device
        model_ = std::make_shared<torch::nn::Module>(base_model_.clone());
        model_->to(device_);

        // Broadcast initial parameters from rank 0
        broadcast_parameters();
    }

    void broadcast_parameters() {
        // Broadcast model parameters from rank 0 to all processes
        for (auto& param : model_->parameters()) {
            if (rank_ == 0) {
                torch::Tensor param_data = param.clone();
                c10d::broadcast(param_data, 0);
                param.copy_(param_data);
            } else {
                c10d::broadcast(param, 0);
            }
        }
    }

    void synchronize_gradients(std::shared_ptr<c10d::ProcessGroupMPI> process_group) {
        // All-reduce gradients across all processes
        std::vector<torch::Tensor> gradients;
        for (auto& param : model_->parameters()) {
            if (param.grad().defined()) {
                gradients.push_back(param.grad());
            }
        }

        if (!gradients.empty()) {
            // Average gradients across all processes
            for (auto& grad : gradients) {
                process_group->allreduce(grad);
                grad.div_(world_size_);
            }
        }
    }

    torch::nn::Module& base_model_;
    std::shared_ptr<torch::nn::Module> model_;
    int world_size_;
    int rank_;
    torch::Device device_;
    size_t batch_size_;
};

// Example usage:
/*
#include <mpi.h>
#include "distributed_data_parallel.h"

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Set device
    torch::Device device(torch::cuda::is_available() && rank < torch::cuda::device_count()
        ? torch::Device(torch::kCUDA, rank)
        : torch::Device(torch::kCPU));

    // Define model
    auto model = torch::nn::Sequential(
        torch::nn::Linear(784, 256),
        torch::nn::ReLU(),
        torch::nn::Linear(256, 10)
    );

    // Create dataset with distributed sampler
    auto dataset = torch::data::datasets::MNIST("./data")
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());

    auto sampler = torch::data::samplers::DistributedRandomSampler(
        dataset.size().value(), world_size, rank);

    auto dataloader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions().batch_size(64).workers(2),
        sampler);

    // Create optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));

    // Create DistributedDataParallel
    DistributedDataParallel ddp(model, world_size, rank, device, 64);

    // Train
    ddp.train(*dataloader, optimizer, 10);

    // Cleanup
    MPI_Finalize();
    return 0;
}
*/