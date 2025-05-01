#include <torch/torch.h>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <memory>
#include <iostream>
#include <chrono>

// Custom neural network module inheriting from torch::nn::Cloneable
struct CustomNet : torch::nn::Cloneable<CustomNet> {
    CustomNet() {
        reset();
    }

    void reset() override {
        fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(784, 256).bias(true)));
        fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(256, 10).bias(true)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

// Optimized DataParallel class for multi-GPU training
class DataParallel {
public:
    DataParallel(std::shared_ptr<torch::nn::Module> model,
                 const std::vector<torch::Device>& devices,
                 size_t batch_size)
        : base_model_(model),
          devices_(devices),
          batch_size_(batch_size) {
        if (devices_.empty()) {
            throw std::runtime_error("No devices specified for DataParallel");
        }
        if (batch_size_ % devices_.size() != 0) {
            throw std::runtime_error("Batch size must be divisible by number of devices");
        }
        initialize();
    }

    template<typename DataLoader>
    void train(DataLoader& dataloader, torch::optim::Optimizer& optimizer, size_t epochs) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            std::vector<std::thread> training_threads;
            std::vector<std::queue<std::tuple<torch::Tensor, torch::Tensor>>> batch_queues(devices_.size());
            std::vector<std::mutex> queue_mutexes(devices_.size());
            std::vector<size_t> batch_counts(devices_.size(), 0);

            // Data distribution thread for each device
            std::vector<std::thread> data_threads;
            for (size_t i = 0; i < devices_.size(); ++i) {
                data_threads.emplace_back([&, i]() {
                    size_t batch_idx = 0;
                    for (auto& batch : dataloader) {
                        if (batch_idx % devices_.size() == i) { // Distribute batches round-robin
                            auto data = batch.data.to(devices_[i], /*non_blocking=*/true);
                            auto target = batch.target.to(devices_[i], /*non_blocking=*/true);
                            {
                                std::lock_guard<std::mutex> lock(queue_mutexes[i]);
                                batch_queues[i].push(std::make_tuple(data, target));
                                batch_counts[i]++;
                            }
                        }
                        batch_idx++;
                    }
                });
            }

            // Training threads for each device
            for (size_t i = 0; i < devices_.size(); ++i) {
                training_threads.emplace_back([&, i]() {
                    auto model = models_[i];
                    model->train();
                    while (true) {
                        torch::Tensor data, target;
                        {
                            std::lock_guard<std::mutex> lock(queue_mutexes[i]);
                            if (batch_queues[i].empty()) {
                                if (batch_counts[i] == 0) break;
                                continue;
                            }
                            std::tie(data, target) = batch_queues[i].front();
                            batch_queues[i].pop();
                            batch_counts[i]--;
                        }

                        // Forward pass
                        auto output = model->as<CustomNet>()->forward(data);
                        auto loss = torch::nn::functional::cross_entropy(output, target);

                        // Backward pass
                        optimizer.zero_grad();
                        loss.backward();

                        // Synchronize gradients
                        synchronize_gradients(i);
                    }
                });
            }

            // Update parameters in main thread after all threads complete
            for (auto& thread : training_threads) {
                thread.join();
            }
            for (auto& thread : data_threads) {
                thread.join();
            }

            // Perform optimizer step
            optimizer.step();
            broadcast_parameters();

            // Synchronize CUDA devices
            torch::cuda::synchronize();

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << "Epoch " << epoch + 1 << " completed in " << duration << " ms\n";
            start_time = end_time;
        }
    }

private:
    void initialize() {
        // Replicate model to all devices
        for (const auto& device : devices_) {
            auto model = base_model_->clone();
            model->to(device, /*non_blocking=*/true);
            models_.push_back(model);
        }
    }

    void synchronize_gradients(size_t device_idx) {
        std::lock_guard<std::mutex> lock(grad_mutex_);
        if (device_idx == 0) {
            // Aggregate gradients from all devices
            for (size_t i = 1; i < models_.size(); ++i) {
                auto params_i = models_[i]->parameters();
                auto params_0 = models_[0]->parameters();
                for (size_t j = 0; j < params_i.size(); ++j) {
                    if (params_i[j].grad().defined()) {
                        auto grad = params_i[j].grad().to(devices_[0], /*non_blocking=*/true).contiguous();
                        if (params_0[j].grad().defined()) {
                            auto new_grad = params_0[j].grad().clone().to(devices_[0], /*non_blocking=*/true).contiguous();
                            new_grad += grad;
                            params_0[j].mutable_grad() = new_grad;
                        } else {
                            params_0[j].mutable_grad() = grad.clone();
                        }
                    }
                }
            }
        }
    }

    void broadcast_parameters() {
        auto params_0 = models_[0]->parameters();
        for (size_t i = 1; i < models_.size(); ++i) {
            auto params_i = models_[i]->parameters();
            for (size_t j = 0; j < params_i.size(); ++j) {
                params_i[j].copy_(params_0[j].to(devices_[i], /*non_blocking=*/true));
            }
        }
    }

    std::shared_ptr<torch::nn::Module> base_model_;
    std::vector<torch::Device> devices_;
    std::vector<std::shared_ptr<torch::nn::Module>> models_;
    size_t batch_size_;
    std::mutex grad_mutex_;
};

// Main program
int main() {
    // Ensure CUDA is available
    if (!torch::cuda::is_available() || torch::cuda::device_count() < 2) {
        std::cerr << "Need at least 2 CUDA devices. Exiting." << std::endl;
        return 1;
    }

    // Define custom model
    auto model = std::make_shared<CustomNet>();

    // Define devices (2 GPUs)
    std::vector<torch::Device> devices = {
        torch::Device(torch::kCUDA, 0),
        torch::Device(torch::kCUDA, 1)
    };

    // Create DataParallel (batch size divisible by 2)
    DataParallel dp(model, devices, 128); // Larger batch size for better GPU utilization

    // Create dataset and dataloader
    auto dataset = torch::data::datasets::MNIST("/home/kami/Documents/datasets/MNIST/raw/")
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());
    auto dataloader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions().batch_size(128).workers(4));

    // Create optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));
    auto start = std::chrono::high_resolution_clock::now();
    // Train
    dp.train(*dataloader, optimizer, 50);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << duration << " s\n";


    return 0;
}