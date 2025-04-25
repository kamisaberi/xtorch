#pragma once

#include <torch/torch.h>
#include <iostream>
#include <filesystem>

#include <memory>  // For std::shared_ptr
#include <string>  // For std::string
#include "../data_loaders/data_loader.h"
#include "../models/base.h"
#include "../datasets/base/base.h"


// Forward declarations (assume these classes exist elsewhere)
//class Optimizer;
//class LossFunction;

namespace xt {

//    template <typename Dataset>
//    void check_dataset_type(const Dataset& dataset) {
//        if constexpr (std::is_same_v<Dataset, xt::data::datasets::BaseDataset>) {
//            std::cout << "The object is a MNIST dataset" << std::endl;
//        } else if constexpr (std::is_same_v<Dataset, torch::data::datasets::MapDataset<xt::data::datasets::BaseDataset, torch::data::transforms::Stack<>>>) {
//            std::cout << "The object is a transformed MNIST dataset" << std::endl;
//        } else {
//            std::cout << "The object is of an unknown type" << std::endl;
//        }
//    }



    class Trainer {
    public:
        // Default constructor
        Trainer();

        // Setter methods with fluent interface
        Trainer& set_max_epochs(int maxEpochs);
        Trainer& set_optimizer(torch::optim::Optimizer *optimizer);
        Trainer& set_loss_fn(std::function<torch::Tensor(torch::Tensor, torch::Tensor)> lossFn);
        Trainer& enable_checkpoint(const std::string& path, int interval);
        template <typename Dataset>
//        void fit(torch::ext::models::BaseModel *model , xt::DataLoader<Dataset>&  train_loader);
//        template <typename Dataset>
        void fit(xt::models::BaseModel *model , xt::DataLoader<Dataset>&  train_loader) {

            torch::Device device(torch::kCPU);
            model->to(device);
            model->train();
            for (size_t epoch = 0; epoch != this->max_epochs_; ++epoch) {
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
                    loss = this->loss_fn_(output, targets);
                    loss.backward();
                    this->optimizer_->step();
                    a++;

                }
                cout << "interval: " << a << endl;
            }
        }

        template <typename Dataset>
        void fit(xt::models::BaseModel *model , xt::DataLoader<Dataset>&  train_loader , torch::Device device) {

            model->to(device);
            model->train();
            for (size_t epoch = 0; epoch != this->max_epochs_; ++epoch) {
                cout << "epoch: " << epoch << endl;
                int a = 1;
                for (auto& batch : train_loader) {
                    torch::Tensor data, targets;
                    data = batch.data;
                    targets = batch.target;
                    data = data.to(device);
                    targets = targets.to(device);
                    this->optimizer_->zero_grad();
                    torch::Tensor output;
                    output = model->forward(data);
                    torch::Tensor loss;
                    loss = this->loss_fn_(output, targets);
                    loss.backward();
                    this->optimizer_->step();
                    a++;

                }
                cout << "interval: " << a << endl;
            }
        }


    private:
        int max_epochs_;                         // Maximum number of training epochs
        torch::optim::Optimizer *optimizer_;  // Optimizer object
        std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_fn_;
//        std::shared_ptr<torch::nn::CrossEntropyLoss> lossFn_;  // Loss function object
        bool checkpoint_enabled_;                // Flag for checkpointing status
        std::string checkpoint_path_;            // Path for saving checkpoints
        int checkpoint_interval_;                // Interval for checkpointing
    };

} // namespace xt



