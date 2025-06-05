#pragma once

#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <memory> // For std::shared_ptr, std::unique_ptr
#include <string>
#include <vector>
#include <functional> // For std::function
#include <limits>     // For std::numeric_limits

#include "../data_loaders/extended_data_loader.h" // Your DataLoader
#include "../base/base.h"     // For xt::Module
#include "callback.h"              // For xt::Callback

namespace xt {

    class Trainer {
    public:
        Trainer();

        // Configuration
        Trainer& set_max_epochs(int max_epochs);
        Trainer& set_optimizer(torch::optim::Optimizer& optimizer); // Non-owning
        Trainer& set_loss_fn(std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> loss_fn);
        Trainer& set_lr_scheduler(std::shared_ptr<torch::optim::LRScheduler> scheduler); // Can be nullptr
        Trainer& add_callback(std::shared_ptr<Callback> callback);
        Trainer& set_gradient_accumulation_steps(int steps);

        // Checkpointing
        Trainer& enable_checkpointing(const std::string& save_dir,
                                      const std::string& metric_to_monitor = "val_loss",
                                      const std::string& mode = "min"); // "min" or "max"

        // Main training method
        void fit(xt::Module& model,
                 xt::dataloaders::ExtendedDataLoader& train_loader,
                 xt::dataloaders::ExtendedDataLoader* val_loader = nullptr, // Optional validation loader
                 torch::Device device = torch::Device(torch::kCPU));

        // Allow callbacks to access some trainer state (use with caution)
        const xt::Module* get_model_ptr() const { return model_ptr_; }
        const torch::optim::Optimizer* get_optimizer_ptr() const { return optimizer_ptr_; }
        int get_current_epoch() const { return current_epoch_; }


    private:
        void train_epoch(xt::dataloaders::ExtendedDataLoader& train_loader, torch::Device device);
        EpochEndState validate_epoch(xt::dataloaders::ExtendedDataLoader& val_loader, torch::Device device);
        void save_checkpoint(const std::string& reason = "best_model");


        // Core components
        xt::Module* model_ptr_; // Non-owning pointer to the model being trained
        torch::optim::Optimizer* optimizer_ptr_; // Non-owning
        std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> loss_fn_;
        std::shared_ptr<torch::optim::LRScheduler> lr_scheduler_;

        // Training parameters
        int max_epochs_;
        int current_epoch_; // To track current epoch
        torch::Device current_device_; // Device used for current training
        int grad_accumulation_steps_;

        // Callbacks
        std::vector<std::shared_ptr<Callback>> callbacks_;

        // Checkpointing
        bool checkpointing_enabled_;
        std::string checkpoint_save_dir_;
        std::string checkpoint_metric_monitor_; // e.g., "val_loss"
        std::string checkpoint_mode_;           // "min" or "max"
        double best_metric_value_;
    };

} // namespace xt