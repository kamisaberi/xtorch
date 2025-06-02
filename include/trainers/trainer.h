#pragma once

#include <torch/torch.h>      // LibTorch for tensor operations and neural networks
#include <iostream>           // Standard input/output for logging
#include <filesystem>         // Filesystem operations (e.g., for checkpoint paths)
#include <memory>             // For std::shared_ptr
#include <string>             // For std::string
#include "include/data_loaders/data_loaders.h"  // Custom data loader for datasets
#include "include/datasets/common.h"        // Base dataset class for data handling
#include "include/base/base.h"
namespace xt {
    class Trainer {
    public:
        Trainer();
        Trainer& set_max_epochs(int maxEpochs);
        Trainer& set_optimizer(torch::optim::Optimizer *optimizer);
        Trainer& set_loss_fn(std::function<torch::Tensor(torch::Tensor, torch::Tensor)> lossFn);
        Trainer& enable_checkpoint(const std::string& path, int interval);
        template <typename Dataset>
        void fit(xt::Module *model, xt::DataLoader<Dataset>& train_loader) {
            torch::Device device(torch::kCPU);
            model->to(device);
            model->train();
            for (size_t epoch = 0; epoch != this->max_epochs_; ++epoch) {
                std::cout << "epoch: " << epoch << std::endl; // Log current epoch
                int a = 1; // Counter for batches (debugging or logging)
                for (auto& batch : train_loader) {
                    torch::Tensor data, targets;
                    data = batch.data;
                    targets = batch.target;
                    this->optimizer_->zero_grad();
                    torch::Tensor output;
                    output = std::any_cast<torch::Tensor>(model->forward({data}));
                    torch::Tensor loss;
                    // Compute loss using the specified loss function
                    loss = this->loss_fn_(output, targets);
                    // Backward pass to compute gradients
                    loss.backward();
                    this->optimizer_->step();
                    a++; // Increment batch counter
                }
                std::cout << "interval: " << a << std::endl;
            }
        }

        template <typename Dataset>
        void fit(xt::Module *model, xt::DataLoader<Dataset>& train_loader, torch::Device device) {
            model->to(device);
            model->train();
            for (size_t epoch = 0; epoch != this->max_epochs_; ++epoch) {
                std::cout << "epoch: " << epoch << std::endl; // Log current epoch
                int a = 1; // Counter for batches (debugging or logging)
                for (auto& batch : train_loader) {
                    torch::Tensor data, targets;
                    data = batch.data;
                    targets = batch.target;
                    data = data.to(device);
                    targets = targets.to(device);
                    this->optimizer_->zero_grad();
                    torch::Tensor output;
                    output = std::any_cast<torch::Tensor>(model->forward({data}));
                    torch::Tensor loss;
                    loss = this->loss_fn_(output, targets);
                    loss.backward();
                    this->optimizer_->step();
                    a++; // Increment batch counter
                }
                std::cout << "interval: " << a << std::endl;
            }
        }

    private:
        int max_epochs_; // Maximum number of training epochs
        torch::optim::Optimizer *optimizer_; // Pointer to the optimizer (e.g., Adam, SGD)
        // Loss function as a std::function taking output and target tensors
        std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_fn_;
        bool checkpoint_enabled_; // Flag indicating whether checkpointing is enabled
        std::string checkpoint_path_; // Directory path for saving model checkpoints
        int checkpoint_interval_; // Number of epochs between saving checkpoints
    };

} // namespace xt