#pragma once

// Include necessary headers for LibTorch, standard I/O, filesystem operations, and memory management
#include <torch/torch.h>      // LibTorch for tensor operations and neural networks
#include <iostream>           // Standard input/output for logging
#include <filesystem>         // Filesystem operations (e.g., for checkpoint paths)
#include <memory>             // For std::shared_ptr
#include <string>             // For std::string
#include "../data_loaders/data_loader.h"  // Custom data loader for datasets
#include "../models/base.h"               // Base model class for neural networks
#include "../datasets/base/base.h"        // Base dataset class for data handling

namespace xt {
// Namespace for the xTorch project, encapsulating training-related functionality

    // Trainer class for managing the training process of deep learning models
    class Trainer {
    public:
        // Default constructor
        // Initializes the Trainer with default values for max_epochs_, optimizer_, etc.
        Trainer();

        // Setter methods with fluent interface (return *this for method chaining)

        // Sets the maximum number of training epochs
        // @param maxEpochs Number of epochs to train
        // @return Reference to this Trainer for chaining
        Trainer& set_max_epochs(int maxEpochs);

        // Sets the optimizer for training (e.g., Adam, SGD)
        // @param optimizer Pointer to a torch::optim::Optimizer object
        // @return Reference to this Trainer for chaining
        Trainer& set_optimizer(torch::optim::Optimizer *optimizer);

        // Sets the loss function as a std::function that takes output and target tensors
        // @param lossFn Function that computes the loss (e.g., torch::nll_loss)
        // @return Reference to this Trainer for chaining
        Trainer& set_loss_fn(std::function<torch::Tensor(torch::Tensor, torch::Tensor)> lossFn);

        // Enables model checkpointing to save model state at specified intervals
        // @param path Directory path to save checkpoints
        // @param interval Number of epochs between checkpoints
        // @return Reference to this Trainer for chaining
        Trainer& enable_checkpoint(const std::string& path, int interval);

        // Trains a model on a dataset using the specified data loader (CPU version)
        // @tparam Dataset Type of the dataset (e.g., xt::data::datasets::BaseDataset)
        // @param model Pointer to a model derived from xt::models::BaseModel
        // @param train_loader DataLoader providing batches of training data
        // @note Assumes CPU device; moves model and data to CPU
        template <typename Dataset>
        void fit(xt::models::BaseModel *model, xt::DataLoader<Dataset>& train_loader) {
            // Set device to CPU
            torch::Device device(torch::kCPU);
            // Move model to CPU and set to training mode
            model->to(device);
            model->train();
            // Iterate over specified number of epochs
            for (size_t epoch = 0; epoch != this->max_epochs_; ++epoch) {
                std::cout << "epoch: " << epoch << std::endl; // Log current epoch
                int a = 1; // Counter for batches (debugging or logging)
                // Iterate over batches in the data loader
                for (auto& batch : train_loader) {
                    torch::Tensor data, targets;
                    // Extract data and targets from batch
                    data = batch.data;
                    targets = batch.target;
                    // Zero out gradients before forward pass
                    this->optimizer_->zero_grad();
                    torch::Tensor output;
                    // Forward pass through the model
                    output = model->forward(data);
                    torch::Tensor loss;
                    // Compute loss using the specified loss function
                    loss = this->loss_fn_(output, targets);
                    // Backward pass to compute gradients
                    loss.backward();
                    // Update model parameters using the optimizer
                    this->optimizer_->step();
                    a++; // Increment batch counter
                }
                // Log the number of batches processed (interval is misleading; should be batch count)
                std::cout << "interval: " << a << std::endl;
            }
        }

        // Trains a model on a dataset using the specified data loader and device (e.g., CPU or CUDA)
        // @tparam Dataset Type of the dataset (e.g., xt::data::datasets::BaseDataset)
        // @param model Pointer to a model derived from xt::models::BaseModel
        // @param train_loader DataLoader providing batches of training data
        // @param device Target device (e.g., torch::kCUDA, torch::kCPU)
        // @note Moves model and data to the specified device
        template <typename Dataset>
        void fit(xt::models::BaseModel *model, xt::DataLoader<Dataset>& train_loader, torch::Device device) {
            // Move model to the specified device and set to training mode
            model->to(device);
            model->train();
            // Iterate over specified number of epochs
            for (size_t epoch = 0; epoch != this->max_epochs_; ++epoch) {
                std::cout << "epoch: " << epoch << std::endl; // Log current epoch
                int a = 1; // Counter for batches (debugging or logging)
                // Iterate over batches in the data loader
                for (auto& batch : train_loader) {
                    torch::Tensor data, targets;
                    // Extract data and targets from batch
                    data = batch.data;
                    targets = batch.target;
                    // Move data and targets to the specified device
                    data = data.to(device);
                    targets = targets.to(device);
                    // Zero out gradients before forward pass
                    this->optimizer_->zero_grad();
                    torch::Tensor output;
                    // Forward pass through the model
                    output = model->forward(data);
                    torch::Tensor loss;
                    // Compute loss using the specified loss function
                    loss = this->loss_fn_(output, targets);
                    // Backward pass to compute gradients
                    loss.backward();
                    // Update model parameters using the optimizer
                    this->optimizer_->step();
                    a++; // Increment batch counter
                }
                // Log the number of batches processed (interval is misleading; should be batch count)
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