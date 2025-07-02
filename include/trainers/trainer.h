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

namespace xt
{
    class Trainer
    {
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
        std::string checkpoint_mode_; // "min" or "max"
        double best_metric_value_;


        // =================================================================================
        // TEMPLATE IMPLEMENTATIONS - INSIDE THE HEADER BUT OUTSIDE THE CLASS DEFINITION
        // This is the standard and correct way to do it.
        // =================================================================================
    public:
        template <typename TrainLoaderT, typename ValLoaderT>
        void fit(xt::Module& model, // <<< FIX WAS HERE
                 TrainLoaderT& train_loader,
                 ValLoaderT* val_loader,
                 torch::Device device)
        {
            if (!optimizer_ptr_) throw std::runtime_error("Optimizer not set.");
            if (!loss_fn_) throw std::runtime_error("Loss function not set.");

            model_ptr_ = &model;
            current_device_ = device;
            model_ptr_->to(current_device_);

            for (const auto& cb : callbacks_) cb->on_train_begin();

            for (current_epoch_ = 1; current_epoch_ <= max_epochs_; ++current_epoch_)
            {
                for (const auto& cb : callbacks_) cb->on_epoch_begin(current_epoch_);

                model_ptr_->train();
                train_epoch(train_loader, current_device_);

                EpochEndState epoch_state;
                epoch_state.epoch = current_epoch_;

                if (val_loader)
                {
                    model_ptr_->eval();
                    EpochEndState val_epoch_state = validate_epoch(*val_loader, current_device_);
                    epoch_state.val_loss = val_epoch_state.val_loss;

                    if (checkpointing_enabled_ && checkpoint_metric_monitor_ == "val_loss")
                    {
                        bool improved = (checkpoint_mode_ == "min" && epoch_state.val_loss < best_metric_value_) ||
                            (checkpoint_mode_ == "max" && epoch_state.val_loss > best_metric_value_);
                        if (improved)
                        {
                            best_metric_value_ = epoch_state.val_loss;
                            save_checkpoint();
                        }
                    }
                }

                for (const auto& cb : callbacks_) cb->on_epoch_end(epoch_state);

                if (lr_scheduler_)
                {
                    if (auto* plateau_scheduler = dynamic_cast<torch::optim::ReduceLROnPlateauScheduler*>(lr_scheduler_.
                        get()))
                    {
                        if (val_loader) plateau_scheduler->step(epoch_state.val_loss);
                    }
                    else
                    {
                        lr_scheduler_->step();
                    }
                }
            }
            for (const auto& cb : callbacks_) cb->on_train_end();
            model_ptr_ = nullptr;
        }

        template <typename LoaderT>
        void train_epoch(LoaderT& train_loader, torch::Device device) // <<< FIX WAS HERE
        {
            size_t batch_idx = 0;
            optimizer_ptr_->zero_grad();

            for (auto& batch_data : train_loader)
            {
                for (const auto& cb : callbacks_) cb->on_batch_begin(current_epoch_, static_cast<int>(batch_idx));

                torch::Tensor data = batch_data.first.to(device);
                torch::Tensor target = batch_data.second.to(device);

                auto output_any = model_ptr_->forward({data});
                torch::Tensor output = std::any_cast<torch::Tensor>(output_any);
                torch::Tensor loss = loss_fn_(output, target);

                if (grad_accumulation_steps_ > 1) { loss = loss / grad_accumulation_steps_; }

                loss.backward();

                if ((batch_idx + 1) % grad_accumulation_steps_ == 0)
                {
                    optimizer_ptr_->step();
                    optimizer_ptr_->zero_grad();
                }

                BatchEndState batch_state(current_epoch_, static_cast<int>(batch_idx),
                                          loss.detach() * grad_accumulation_steps_, output.detach(), target.detach(),
                                          data.size(0));
                for (const auto& cb : callbacks_) cb->on_batch_end(batch_state);

                batch_idx++;
            }
        }

        template <typename LoaderT>
        EpochEndState validate_epoch(LoaderT& val_loader, torch::Device device) // <<< FIX WAS HERE
        {
            torch::NoGradGuard no_grad;
            double running_val_loss = 0.0;
            size_t num_val_samples_processed = 0;

            for (auto& batch_data : val_loader)
            {
                torch::Tensor data = batch_data.first.to(device);
                torch::Tensor target = batch_data.second.to(device);
                std::any output_any = model_ptr_->forward({data});
                auto output = std::any_cast<torch::Tensor>(output_any);
                torch::Tensor loss = loss_fn_(output, target);
                running_val_loss += loss.item<double>() * data.size(0);
                num_val_samples_processed += data.size(0);
            }

            EpochEndState val_results;
            val_results.epoch = current_epoch_;
            val_results.val_loss = (num_val_samples_processed > 0)
                                       ? (running_val_loss / num_val_samples_processed)
                                       : 0.0;
            return val_results;
        }
    };
} // namespace xt
