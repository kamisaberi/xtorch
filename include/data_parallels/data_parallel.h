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
