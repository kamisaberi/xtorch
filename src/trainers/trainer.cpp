#include "include/trainers/trainer.h" // Assuming trainer.h is in the same include path or adjust
#include <any> // For std::any_cast

namespace xt
{
    Trainer::Trainer()
        : max_epochs_(10), // Default to a reasonable number of epochs
          optimizer_ptr_(nullptr),
          loss_fn_(nullptr),
          checkpoint_enabled_(false),
          checkpoint_path_(""),
          checkpoint_interval_(1) // Default checkpoint interval
    {
    }

    Trainer& Trainer::set_max_epochs(int max_epochs)
    {
        if (max_epochs <= 0)
        {
            throw std::invalid_argument("Max epochs must be positive.");
        }
        max_epochs_ = max_epochs;
        return *this;
    }

    // Store a pointer to the optimizer. The optimizer's lifetime must be managed externally.
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

    Trainer& Trainer::enable_checkpoint(const std::string& path, int interval)
    {
        if (interval <= 0)
        {
            throw std::invalid_argument("Checkpoint interval must be positive.");
        }
        checkpoint_path_ = path;
        checkpoint_interval_ = interval;
        checkpoint_enabled_ = true;
        // Optionally, create the directory if it doesn't exist
        // if (!std::filesystem::exists(checkpoint_path_)) {
        //     std::filesystem::create_directories(checkpoint_path_);
        // }
        return *this;
    }

    // Private helper for the main training loop
    void Trainer::training_loop(xt::Module& model,
                                xt::datasets::Dataset& train_loader,
                                torch::Device device)
    {
        if (!optimizer_ptr_)
        {
            throw std::runtime_error("Optimizer not set in Trainer.");
        }
        if (!loss_fn_)
        {
            throw std::runtime_error("Loss function not set in Trainer.");
        }

        model.to(device); // Move model to the specified device
        model.train(); // Set the model to training mode

        for (int epoch = 1; epoch <= max_epochs_; ++epoch)
        {
            std::cout << "Epoch " << epoch << "/" << max_epochs_ << std::endl;
            size_t batch_idx = 0;
            double running_loss = 0.0;
            int num_samples_processed = 0;

            // train_loader.begin() will call reset_epoch() on the DataLoader
            for (auto& batch : train_loader)
            {
                // BatchData is std::pair<torch::Tensor, torch::Tensor>
                torch::Tensor data = batch.first.to(device);
                torch::Tensor targets = batch.second.to(device);

                optimizer_ptr_->zero_grad(); // Clear previous gradients

                // Forward pass
                // Assuming model.forward takes std::initializer_list<std::any>
                // and returns std::any which can be cast to torch::Tensor
                std::any output_any = model.forward({data});
                torch::Tensor output = std::any_cast<torch::Tensor>(output_any);

                // Ensure output and targets are compatible for loss calculation
                // For example, if using CrossEntropyLoss, output is raw scores, targets are class indices.
                // If output is, e.g., [N, C] and targets are [N], it's usually fine.
                // If targets are one-hot encoded and loss expects class indices, targets might need .argmax(1)
                // This depends heavily on your model output and loss function.

                torch::Tensor loss = loss_fn_(output, targets);

                loss.backward(); // Compute gradients
                optimizer_ptr_->step(); // Update model parameters

                running_loss += loss.item<double>() * data.size(0); // loss.item() for scalar loss
                num_samples_processed += data.size(0);

                if ((batch_idx + 1) % 10 == 0)
                {
                    // Assuming DataLoader has total_batches_in_epoch_
                    std::cout << "  Batch " << (batch_idx + 1) << " | Loss: " << (running_loss / num_samples_processed)
                        << std::endl;
                }
                batch_idx++;
            }
            double epoch_loss = running_loss / num_samples_processed;
            std::cout << "Epoch " << epoch << " Summary | Average Loss: " << epoch_loss << std::endl;

            // Checkpointing (simplified example)
            // if (checkpoint_enabled_ && (epoch % checkpoint_interval_ == 0)) {
            //     std::string ckpt_path = checkpoint_path_ + "/model_epoch_" + std::to_string(epoch) + ".pt";
            //     torch::save(model, ckpt_path); // Save the whole model
            //     // Or save state dict: torch::save(model.named_parameters(), ckpt_path_state_dict);
            //     std::cout << "Checkpoint saved to " << ckpt_path << std::endl;
            // }
        }
        std::cout << "Training finished." << std::endl;
    }


    // Fit method for CPU training (delegates to the device-specific one)
    void Trainer::fit(xt::Module& model,
                      xt::datasets::Dataset& train_loader)
    {
        training_loop(model, train_loader, torch::Device(torch::kCPU));
    }

    // Fit method for training on a specified device
    void Trainer::fit(xt::Module& model,
                      xt::datasets::Dataset& train_loader,
                      torch::Device device)
    {
        training_loop(model, train_loader, device);
    }
} // namespace xt
