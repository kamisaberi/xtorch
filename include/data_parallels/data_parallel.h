#pragma once
#include <torch/torch.h>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <memory>
#include <iostream>
#include <chrono>
#include <stdexcept>


// Custom neural network module inheriting from torch::nn::Cloneable
struct CustomNet : torch::nn::Cloneable<CustomNet> {
    CustomNet() {
        reset();
    }

    void reset() override {
        fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(784, 256).bias(true)));
        fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(256, 10).bias(true)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};


// Optimized DataParallel class for multi-GPU training
class DataParallel
{
public:
    DataParallel(std::shared_ptr<torch::nn::Module> model, const std::vector<torch::Device>& devices,
                 size_t batch_size);

    template <typename DatasetType>
    void train(DatasetType& dataset, torch::optim::Optimizer& optimizer, size_t epochs);

private:
    void initialize();

    void synchronize_gradients(size_t device_idx);

    void broadcast_parameters();

    std::shared_ptr<torch::nn::Module> base_model_;
    std::vector<torch::Device> devices_;
    std::vector<std::shared_ptr<torch::nn::Module>> models_;
    size_t batch_size_;
    std::mutex grad_mutex_;
};
