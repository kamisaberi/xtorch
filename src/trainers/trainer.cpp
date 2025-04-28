/**
 * @file trainer.cpp
 * @brief Model training utility implementation
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: [Current Date]
 * @version 1.0
 *
 * This file implements the Trainer class which provides:
 * - Configurable model training workflow
 * - Optimizer and loss function management
 * - Training checkpoint functionality
 */

#include "../../include/trainers/trainer.h"

namespace xt {

    /**
     * @brief Constructs a new Trainer object with default values
     *
     * Initializes:
     * - max_epochs_ to 0
     * - optimizer_ to nullptr
     * - loss_fn_ to nullptr
     * - checkpoint_enabled_ to false
     * - checkpoint_path_ to empty string
     * - checkpoint_interval_ to 0
     */
    Trainer::Trainer()
            : max_epochs_(0),
              optimizer_(nullptr),
              loss_fn_(nullptr),
              checkpoint_enabled_(false),
              checkpoint_path_(""),
              checkpoint_interval_(0)
    {
        // Default constructor initializes members
    }

    /**
     * @brief Sets the maximum number of training epochs
     * @param maxEpochs Maximum number of epochs to train
     * @return Trainer& Reference to self for method chaining
     *
     * Example:
     * @code
     * trainer.set_max_epochs(100).set_optimizer(&optim);
     * @endcode
     */
    Trainer& Trainer::set_max_epochs(int maxEpochs) {
        max_epochs_ = maxEpochs;
        return *this;  // Return reference to self for chaining
    }

    /**
     * @brief Sets the optimizer for training
     * @param optimizer Pointer to torch::optim::Optimizer instance
     * @return Trainer& Reference to self for method chaining
     *
     * Note: The Trainer does not take ownership of the optimizer.
     * The caller must ensure the optimizer remains valid during training.
     */
    Trainer& Trainer::set_optimizer(torch::optim::Optimizer *optimizer) {
        optimizer_ = optimizer;  // Take ownership or share
        return *this;
    }

    /**
     * @brief Sets the loss function for training
     * @param lossFn Function that computes loss between predictions and targets
     * @return Trainer& Reference to self for method chaining
     *
     * The loss function should have signature:
     * torch::Tensor(torch::Tensor predictions, torch::Tensor targets)
     */
    Trainer& Trainer::set_loss_fn(std::function<torch::Tensor(torch::Tensor, torch::Tensor)> lossFn) {
        loss_fn_ = lossFn;  // Take ownership or share
        return *this;
    }

    /**
     * @brief Enables model checkpointing during training
     * @param path Directory path to save checkpoints
     * @param interval Save interval in epochs
     * @return Trainer& Reference to self for method chaining
     *
     * Checkpoints will be saved every 'interval' epochs to the specified path.
     * Disabled by default - must be explicitly enabled.
     */
    Trainer& Trainer::enable_checkpoint(const std::string& path, int interval) {
        checkpoint_path_ = path;
        checkpoint_interval_ = interval;
        checkpoint_enabled_ = true;  // Enable checkpointing
        return *this;
    }

    /**
     * @brief Template for model training method (commented out)
     * @tparam Dataset Type of dataset used for training
     * @param model Pointer to the model to train
     * @param train_loader DataLoader for training data
     *
     * This commented implementation shows the intended training workflow:
     * 1. Moves model to device
     * 2. Sets model to training mode
     * 3. Runs epochs loop
     * 4. Processes batches
     * 5. Computes forward pass
     * 6. Calculates loss
     * 7. Backpropagates
     * 8. Updates weights
     *
     * Uncomment and implement as needed for specific use cases.
     */
    //    template <typename Dataset>
    //    void Trainer::fit(torch::ext::models::BaseModel *model , xt::DataLoader<Dataset>&  train_loader) {
    //
    //        torch::Device device(torch::kCPU);
    //        model->to(device);
    //        model->train();
    //        for (size_t epoch = 0; epoch != this->maxEpochs_; ++epoch) {
    //            cout << "epoch: " << epoch << endl;
    //            for (auto& batch : train_loader) {
    //                torch::Tensor data, targets;
    //                data = batch.data;
    //                targets = batch.target;
    //                this->optimizer_->zero_grad();
    //                torch::Tensor output;
    //                output = model->forward(data);
    //                torch::Tensor loss;
    //                loss = this->lossFn_(output, targets);
    ////                loss = torch::nll_loss(output, targets);
    //                loss.backward();
    //                this->optimizer_->step();
    //                //                std::cout << "Epoch: " << epoch << " | Batch: " <<  " | Loss: " << loss.item<float>() <<                            std::endl;
    //
    //                //            }
    //
    //            }
    //        }
    //    }

} // namespace xt

/*
 * Author: Kamran Saberifard
 * Email: kamisaberi@gmail.com
 * GitHub: https://github.com/kamisaberi
 *
 * This implementation is part of the XT Machine Learning Library.
 * Copyright (c) 2023 Kamran Saberifard. All rights reserved.
 *
 * License: MIT
 *
 * Dependencies:
 * - PyTorch C++ API (libtorch)
 * - C++17 or later
 */