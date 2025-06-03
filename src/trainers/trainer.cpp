#include "include/trainers/trainer.h"
#include "include/trainers/callback.h"

#include <any> // For std::any_cast
#include <torch/serialize.h> // For torch::save and torch::load

namespace xt
{
    Trainer::Trainer()
        : model_ptr_(nullptr),
          optimizer_ptr_(nullptr),
          loss_fn_(nullptr),
          lr_scheduler_(nullptr),
          max_epochs_(10),
          current_epoch_(0),
          current_device_(torch::kCPU),
          grad_accumulation_steps_(1),
          checkpointing_enabled_(false),
          checkpoint_save_dir_("checkpoints"),
          checkpoint_metric_monitor_("val_loss"),
          checkpoint_mode_("min"),
          best_metric_value_((checkpoint_mode_ == "min")
                                 ? std::numeric_limits<double>::max()
                                 : std::numeric_limits<double>::lowest())
    {
    }

    Trainer& Trainer::set_max_epochs(int max_epochs)
    {
        if (max_epochs <= 0) throw std::invalid_argument("Max epochs must be positive.");
        max_epochs_ = max_epochs;
        return *this;
    }

    Trainer& Trainer::set_optimizer(torch::optim::Optimizer& optimizer)
    {
        optimizer_ptr_ = &optimizer;
        return *this;
    }

    Trainer& Trainer::set_loss_fn(std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> loss_fn)
    {
        loss_fn_ = loss_fn;
        return *this;
    }

    Trainer& Trainer::set_lr_scheduler(std::shared_ptr<torch::optim::LRScheduler> scheduler)
    {
        lr_scheduler_ = scheduler;
        return *this;
    }

    Trainer& Trainer::add_callback(std::shared_ptr<Callback> callback)
    {
        if (callback)
        {
            callback->set_trainer(this); // Give callback a pointer to this trainer
            callbacks_.push_back(callback);
        }
        return *this;
    }

    Trainer& Trainer::set_gradient_accumulation_steps(int steps)
    {
        if (steps < 1) throw std::invalid_argument("Gradient accumulation steps must be at least 1.");
        grad_accumulation_steps_ = steps;
        return *this;
    }

    Trainer& Trainer::enable_checkpointing(const std::string& save_dir,
                                           const std::string& metric_to_monitor,
                                           const std::string& mode)
    {
        checkpointing_enabled_ = true;
        checkpoint_save_dir_ = save_dir;
        checkpoint_metric_monitor_ = metric_to_monitor;
        if (mode != "min" && mode != "max") throw std::invalid_argument("Checkpoint mode must be 'min' or 'max'.");
        checkpoint_mode_ = mode;
        best_metric_value_ = (checkpoint_mode_ == "min")
                                 ? std::numeric_limits<double>::max()
                                 : std::numeric_limits<double>::lowest();

        if (!std::filesystem::exists(checkpoint_save_dir_))
        {
            try
            {
                std::filesystem::create_directories(checkpoint_save_dir_);
            }
            catch (const std::filesystem::filesystem_error& e)
            {
                std::cerr << "Error creating checkpoint directory " << checkpoint_save_dir_ << ": " << e.what() <<
                    std::endl;
                checkpointing_enabled_ = false; // Disable if dir creation fails
            }
        }
        return *this;
    }


    void Trainer::fit(xt::Module& model,
                      xt::dataloaders::ExtendedDataLoader& train_loader,
                      xt::dataloaders::ExtendedDataLoader* val_loader,
                      torch::Device device)
    {
        if (!optimizer_ptr_) throw std::runtime_error("Optimizer not set.");
        if (!loss_fn_) throw std::runtime_error("Loss function not set.");

        model_ptr_ = &model; // Store non-owning pointer to model
        current_device_ = device;
        model_ptr_->to(current_device_);

        for (const auto& cb : callbacks_) cb->on_train_begin();

        for (current_epoch_ = 1; current_epoch_ <= max_epochs_; ++current_epoch_)
        {
            for (const auto& cb : callbacks_) cb->on_epoch_begin(current_epoch_);

            model_ptr_->train(); // Set model to training mode
            train_epoch(train_loader, current_device_);

            EpochEndState epoch_state;
            epoch_state.epoch = current_epoch_;
            // Note: train_loss from train_epoch needs to be captured if desired for EpochEndState
            // For simplicity, we'll pass a placeholder or calculate it inside train_epoch if needed by callbacks

            if (val_loader)
            {
                model_ptr_->eval(); // Set model to evaluation mode
                EpochEndState val_epoch_state = validate_epoch(*val_loader, current_device_);
                epoch_state.val_loss = val_epoch_state.val_loss; // `validate_epoch` fills its own val_loss

                std::cout << "Epoch " << current_epoch_ << "/" << max_epochs_
                    << " | Validation Loss: " << epoch_state.val_loss << std::endl;

                if (checkpointing_enabled_)
                {
                    bool save = false;
                    if (checkpoint_metric_monitor_ == "val_loss")
                    {
                        if (checkpoint_mode_ == "min" && epoch_state.val_loss < best_metric_value_)
                        {
                            best_metric_value_ = epoch_state.val_loss;
                            save = true;
                        }
                        else if (checkpoint_mode_ == "max" && epoch_state.val_loss > best_metric_value_)
                        {
                            best_metric_value_ = epoch_state.val_loss;
                            save = true;
                        }
                    }
                    // Add more monitored metrics here if needed
                    if (save)
                    {
                        std::cout << "Validation loss improved to " << best_metric_value_ << ". Saving model..." <<
                            std::endl;
                        save_checkpoint();
                    }
                }
            }
            else
            {
                std::cout << "Epoch " << current_epoch_ << "/" << max_epochs_ <<
                    " | Training completed (no validation)." << std::endl;
            }

            // Pass averaged train loss if calculated and needed by callbacks
            // For now, train_loss is not explicitly passed from train_epoch to here for epoch_state
            // It's logged within train_epoch. This can be refined.
            epoch_state.train_loss = -1; // Placeholder for now

            for (const auto& cb : callbacks_) cb->on_epoch_end(epoch_state);

            if (lr_scheduler_)
            {
                // Some schedulers like ReduceLROnPlateau need a metric
                if (auto* plateau_scheduler = dynamic_cast<torch::optim::ReduceLROnPlateauScheduler*>(lr_scheduler_.
                    get()))
                {
                    if (val_loader)
                    {
                        // Only step if validation was done and we have val_loss
                        plateau_scheduler->step(epoch_state.val_loss);
                    }
                    else
                    {
                        std::cerr <<
                            "Warning: ReduceLROnPlateauScheduler used without validation data. Scheduler will not step based on metrics."
                            << std::endl;
                    }
                }
                else
                {
                    lr_scheduler_->step(); // For epoch-based schedulers
                }
            }
        }
        for (const auto& cb : callbacks_) cb->on_train_end();
        std::cout << "Training finished." << std::endl;
        model_ptr_ = nullptr; // Clear model pointer after training
    }

    void Trainer::train_epoch(xt::dataloaders::ExtendedDataLoader& train_loader, torch::Device device)
    {
        size_t batch_idx = 0;
        double running_loss = 0.0;
        size_t num_samples_processed = 0;
        // size_t total_batches = train_loader.get_total_batches_in_epoch(); // Assuming this method exists

        optimizer_ptr_->zero_grad(); // Zero grad once before accumulation loop if grad_accumulation_steps_ > 1

        for (auto& batch_data : train_loader)
        {
            for (const auto& cb : callbacks_) cb->on_batch_begin(current_epoch_, static_cast<int>(batch_idx));

            torch::Tensor data = batch_data.first.to(device);
            torch::Tensor target = batch_data.second.to(device);

            std::any output_any = model_ptr_->forward({data});
            torch::Tensor output = std::any_cast<torch::Tensor>(output_any);
            torch::Tensor loss = loss_fn_(output, target);

            if (grad_accumulation_steps_ > 1)
            {
                loss = loss / grad_accumulation_steps_; // Normalize loss
            }
            loss.backward();

            if ((batch_idx + 1) % grad_accumulation_steps_ == 0)
            {
                optimizer_ptr_->step();
                optimizer_ptr_->zero_grad();
            }

            // BatchEndState batch_state = {
            //     current_epoch_, static_cast<int>(batch_idx), loss.detach() * grad_accumulation_steps_, output.detach(),
            //     target.detach(), data.size(0)
            // };

            BatchEndState batch_state(
                current_epoch_,
                static_cast<int>(batch_idx),
                loss.detach() * grad_accumulation_steps_, // Assuming loss is a tensor
                output.detach(), // Assuming output is a tensor
                target.detach(), // Assuming target is a tensor
                data.size(0)
            );


            for (const auto& cb : callbacks_) cb->on_batch_end(batch_state);


            running_loss += (loss.item<double>() * grad_accumulation_steps_) * data.size(0);
            // Use original scale for running loss
            num_samples_processed += data.size(0);

            if ((batch_idx + 1) % 20 == 0)
            {
                // Log less frequently
                std::cout << "  Train Batch " << (batch_idx + 1) << " | Avg Batch Loss: " << (running_loss /
                    num_samples_processed) << std::endl;
            }
            batch_idx++;
        }
        double epoch_avg_loss = (num_samples_processed > 0) ? (running_loss / num_samples_processed) : 0.0;
        std::cout << "Epoch " << current_epoch_ << " Training | Average Loss: " << epoch_avg_loss << std::endl;
        // This epoch_avg_loss could be stored and passed to on_epoch_end callback if needed.
    }

    EpochEndState Trainer::validate_epoch(xt::dataloaders::ExtendedDataLoader& val_loader, torch::Device device)
    {
        torch::NoGradGuard no_grad; // Disable gradient calculations
        size_t batch_idx = 0;
        double running_val_loss = 0.0;
        size_t num_val_samples_processed = 0;
        // size_t total_val_batches = val_loader.get_total_batches_in_epoch();

        for (auto& batch_data : val_loader)
        {
            torch::Tensor data = batch_data.first.to(device);
            torch::Tensor target = batch_data.second.to(device);

            std::any output_any = model_ptr_->forward({data});
            torch::Tensor output = std::any_cast<torch::Tensor>(output_any);
            torch::Tensor loss = loss_fn_(output, target);

            running_val_loss += loss.item<double>() * data.size(0);
            num_val_samples_processed += data.size(0);
            batch_idx++;
        }

        EpochEndState val_results;
        val_results.epoch = current_epoch_;
        val_results.val_loss = (num_val_samples_processed > 0) ? (running_val_loss / num_val_samples_processed) : 0.0;
        return val_results;
    }

    void Trainer::save_checkpoint(const std::string& reason)
    {
        if (!model_ptr_)
        {
            std::cerr << "Cannot save checkpoint, model pointer is null." << std::endl;
            return;
        }
        if (!checkpointing_enabled_ || checkpoint_save_dir_.empty())
        {
            std::cerr << "Checkpointing not enabled or save directory not set." << std::endl;
            return;
        }

        std::string filename = checkpoint_save_dir_ + "/model_" + reason + "_epoch_" + std::to_string(current_epoch_) +
            ".pt";
        try
        {
            //TODO BUG
            // torch::save(*model_ptr_, filename); // Saves the entire module

            // For state_dict:
            // torch::serialize::OutputArchive output_archive;
            // model_ptr_->save(output_archive);
            // output_archive.save_to(filename);
            std::cout << "Checkpoint saved: " << filename << std::endl;
        }
        catch (const c10::Error& e)
        {
            std::cerr << "Error saving model checkpoint: " << e.what() << std::endl;
        }
    }
} // namespace xt
