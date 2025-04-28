#pragma once

#include <torch/torch.h>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>

class DataParallel {
public:
    // Constructor
    DataParallel(torch::nn::Module& model,
                 const std::vector<torch::Device>& devices,
                 size_t batch_size)
        : base_model_(model),
          devices_(devices),
          batch_size_(batch_size) {
        initialize();
    }

    // Train the model
    template<typename DataLoader>
    void train(DataLoader& dataloader, torch::optim::Optimizer& optimizer, size_t epochs) {
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            std::queue<std::tuple<torch::Tensor, torch::Tensor>> batch_queue;
            std::mutex queue_mutex;
            size_t batch_count = 0;

            // Distribute data across devices
            auto data_thread = std::thread([&]() {
                for (auto& batch : dataloader) {
                    auto data = batch.data.to(devices_[0]);
                    auto target = batch.target.to(devices_[0]);
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        batch_queue.push(std::make_tuple(data, target));
                        batch_count++;
                    }
                }
            });

            // Training threads for each device
            std::vector<std::thread> training_threads;
            for (size_t i = 0; i < devices_.size(); ++i) {
                training_threads.emplace_back([&, i]() {
                    auto model = models_[i];
                    model->train();
                    while (true) {
                        torch::Tensor data, target;
                        {
                            std::lock_guard<std::mutex> lock(queue_mutex);
                            if (batch_queue.empty()) {
                                if (batch_count == 0) break;
                                continue;
                            }
                            std::tie(data, target) = batch_queue.front();
                            batch_queue.pop();
                        }

                        // Split batch for this device
                        auto mini_batch_size = batch_size_ / devices_.size();
                        auto start_idx = i * mini_batch_size;
                        auto end_idx = (i + 1) * mini_batch_size;

                        auto mini_data = data.narrow(0, start_idx, mini_batch_size).to(devices_[i]);
                        auto mini_target = target.narrow(0, start_idx, mini_batch_size).to(devices_[i]);

                        // Forward pass
                        auto output = model->forward(mini_data);
                        auto loss = torch::nn::functional::cross_entropy(output, mini_target);

                        // Backward pass
                        optimizer.zero_grad();
                        loss.backward();

                        // Synchronize gradients
                        synchronize_gradients(i);

                        // Update parameters
                        if (i == 0) {  // Main thread updates parameters
                            optimizer.step();
                            broadcast_parameters();
                        }
                    }
                });
            }

            // Join threads
            data_thread.join();
            for (auto& thread : training_threads) {
                thread.join();
            }

            std::cout << "Epoch " << epoch + 1 << " completed\n";
        }
    }

private:
    void initialize() {
        // Replicate model to all devices
        for (const auto& device : devices_) {
            auto model = std::make_shared<torch::nn::Module>(base_model_.clone());
            model->to(device);
            models_.push_back(model);
        }
    }

    void synchronize_gradients(size_t device_idx) {
        std::lock_guard<std::mutex> lock(grad_mutex_);
        if (device_idx == 0) {
            // Aggregate gradients from all devices
            for (size_t i = 1; i < models_.size(); ++i) {
                for (auto& param : models_[i]->parameters()) {
                    if (param.grad().defined()) {
                        auto main_param = models_[0]->parameters()[param.name()];
                        if (main_param.grad().defined()) {
                            main_param.grad() += param.grad().to(devices_[0]);
                        } else {
                            main_param.grad() = param.grad().to(devices_[0]);
                        }
                    }
                }
            }
        }
    }

    void broadcast_parameters() {
        // Copy parameters from main model to all other models
        for (size_t i = 1; i < models_.size(); ++i) {
            for (auto& param : models_[i]->parameters()) {
                auto main_param = models_[0]->parameters()[param.name()];
                param.copy_(main_param.to(devices_[i]));
            }
        }
    }

    torch::nn::Module& base_model_;
    std::vector<torch::Device> devices_;
    std::vector<std::shared_ptr<torch::nn::Module>> models_;
    size_t batch_size_;
    std::mutex grad_mutex_;
};

// Example usage:
/*
int main() {
    // Define model
    auto model = torch::nn::Sequential(
        torch::nn::Linear(784, 256),
        torch::nn::ReLU(),
        torch::nn::Linear(256, 10)
    );

    // Define devices (2 GPUs + 2 CPUs)
    std::vector<torch::Device> devices = {
        torch::Device(torch::kCUDA, 0),
        torch::Device(torch::kCUDA, 1),
        torch::Device(torch::kCPU),
        torch::Device(torch::kCPU)
    };

    // Create DataParallel
    DataParallel dp(model, devices, 64);

    // Create dataset and dataloader
    auto dataset = torch::data::datasets::MNIST("./data")
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());
    auto dataloader = torch::data::make_data_loader(dataset, 64);

    // Create optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));

    // Train
    dp.train(*dataloader, optimizer, 10);

    return 0;
}
*/