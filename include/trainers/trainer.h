#pragma once

#include <torch/torch.h>
#include <iostream>
#include <filesystem> // If you use checkpointing features later
#include <memory>
#include <string>
#include <functional> // For std::function
#include "include/base/base.h"     // Assuming xt::Module is defined here or via common.h
#include "include/data_loaders/data_loaders.h"     // Assuming xt::Module is defined here or via common.h

// Forward declaration for xt::Module if not fully included by base.h
// namespace xt { class Module; }


namespace xt
{
    class Trainer
    {
    public:
        Trainer();

        // Setter methods (fluent interface)
        Trainer& set_max_epochs(int max_epochs);
        Trainer& set_optimizer(torch::optim::Optimizer& optimizer); // Pass by reference
        Trainer& set_loss_fn(std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> loss_fn);
        Trainer& enable_checkpoint(const std::string& path, int interval); // For future use

        // Fit method for CPU training
        void fit(xt::Module& model, // Pass model by reference
                 xt::dataloaders::ExtendedDataLoader& train_loader); // Use the new DataLoader

        // Fit method for training on a specified device (CPU or GPU)
        void fit(xt::Module& model, // Pass model by reference
                 xt::dataloaders::ExtendedDataLoader& train_loader, // Use the new DataLoader
                 torch::Device device);

    private:
        int max_epochs_;
        torch::optim::Optimizer* optimizer_ptr_; // Store as a pointer
        std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> loss_fn_;

        // Checkpointing members (for future expansion)
        bool checkpoint_enabled_;
        std::string checkpoint_path_;
        int checkpoint_interval_;

        // Helper for the training loop logic
        void training_loop(xt::Module& model,
                           xt::dataloaders::ExtendedDataLoader& train_loader,
                           torch::Device device);
    };
} // namespace xt
