#pragma once

#include <torch/torch.h>
#include <iostream>
#include <filesystem>

#include <memory>  // For std::shared_ptr
#include <string>  // For std::string
#include "../data-loaders/data-loader.h"
#include "../models/base.h"



// Forward declarations (assume these classes exist elsewhere)
//class Optimizer;
//class LossFunction;

namespace xt {

    class Trainer {
    public:
        // Default constructor
        Trainer();

        // Setter methods with fluent interface
        Trainer& setMaxEpochs(int maxEpochs);
        Trainer& setOptimizer(torch::optim::Optimizer *optimizer);
        Trainer& setLossFn(std::function<torch::Tensor(torch::Tensor, torch::Tensor)> lossFn);
        Trainer& enableCheckpoint(const std::string& path, int interval);
        template <typename Dataset>
//        void fit(torch::ext::models::BaseModel *model , xt::DataLoader<Dataset>&  train_loader);
//        template <typename Dataset>
        void fit(xt::models::BaseModel *model , xt::DataLoader<Dataset>&  train_loader) {

            torch::Device device(torch::kCPU);
            model->to(device);
            model->train();
            for (size_t epoch = 0; epoch != this->maxEpochs_; ++epoch) {
                cout << "epoch: " << epoch << endl;
                int a = 1;
                for (auto& batch : train_loader) {
                    torch::Tensor data, targets;
                    data = batch.data;
                    targets = batch.target;
                    this->optimizer_->zero_grad();
                    torch::Tensor output;
                    output = model->forward(data);
                    torch::Tensor loss;
                    loss = this->lossFn_(output, targets);
                    loss.backward();
                    this->optimizer_->step();
                    a++;

                }
                cout << "interval: " << a << endl;
            }
        }


    private:
        int maxEpochs_;                         // Maximum number of training epochs
        torch::optim::Optimizer *optimizer_;  // Optimizer object
        std::function<torch::Tensor(torch::Tensor, torch::Tensor)> lossFn_;
//        std::shared_ptr<torch::nn::CrossEntropyLoss> lossFn_;  // Loss function object
        bool checkpointEnabled_;                // Flag for checkpointing status
        std::string checkpointPath_;            // Path for saving checkpoints
        int checkpointInterval_;                // Interval for checkpointing
    };

} // namespace xt



