#include <torch/torch.h>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <memory>
#include <iostream>
#include <chrono>
#include <stdexcept>

// Custom transform to flatten MNIST images
struct Flatten : public torch::data::transforms::TensorTransform<torch::Tensor> {
    torch::Tensor operator()(torch::Tensor input) {
        // Log input shape for debugging
        // std::cout << "Flatten input shape: " << input.sizes() << std::endl;
        // Input: [1, 28, 28] (single sample) -> Output: [784]
        if (input.dim() >= 3) {
            return input.view({-1});
        }
        return input;
    }
};

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

    template <typename DatasetType>
    void train(DatasetType& dataset, torch::optim::Optimizer& optimizer, size_t epochs) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            try {
                // Create fresh DataLoader for each epoch
                auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                    dataset, torch::data::DataLoaderOptions().batch_size(batch_size_).workers(4));

                std::vector<std::queue<std::tuple<torch::Tensor, torch::Tensor>>> batch_queues(devices_.size());
                std::vector<std::mutex> queue_mutexes(devices_.size());
                std::vector<size_t> batch_counts(devices_.size(), 0);
                std::atomic<bool> data_loading_done{false};
                bool data_thread_error = false;

                // Single data distribution thread
                std::thread data_thread([&]() {
                    try {
                        size_t device_idx = 0;
                        // std::cout << "Data thread started for epoch " << epoch + 1 << std::endl;
                        for (auto& batch : *dataloader) {
                            auto data = batch.data;
                            auto target = batch.target;

                            // Log shape after stacking
                            // std::cout << "Batch data shape after stacking: " << data.sizes() << std::endl;

                            // Verify data shape
                            if (data.dim() != 2 || data.size(1) != 784) {
                                std::cerr << "Invalid data shape: " << data.sizes() << std::endl;
                                throw std::runtime_error("Data shape mismatch");
                            }
                            // std::cout << "Processing batch for device " << device_idx << ", shape: " << data.sizes() << std::endl;

                            data = data.to(devices_[device_idx], /*non_blocking=*/true);
                            target = target.to(devices_[device_idx], /*non_blocking=*/true);
                            {
                                std::lock_guard<std::mutex> lock(queue_mutexes[device_idx]);
                                batch_queues[device_idx].push(std::make_tuple(data, target));
                                batch_counts[device_idx]++;
                            }
                            device_idx = (device_idx + 1) % devices_.size();
                        }
                        data_loading_done = true;
                        // std::cout << "Data loading completed for epoch " << epoch + 1 << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Error in data thread: " << e.what() << std::endl;
                        data_loading_done = true;
                        data_thread_error = true;
                    }
                });

                // Training threads for each device
                std::vector<std::thread> training_threads;
                for (size_t i = 0; i < devices_.size(); ++i) {
                    training_threads.emplace_back([&, i]() {
                        try {
                            auto model = models_[i];
                            model->train();
                            // std::cout << "Training thread " << i << " started on " << devices_[i] << std::endl;
                            size_t batches_processed = 0;
                            while (true) {
                                torch::Tensor data, target;
                                bool has_data = false;
                                {
                                    std::lock_guard<std::mutex> lock(queue_mutexes[i]);
                                    if (!batch_queues[i].empty()) {
                                        std::tie(data, target) = batch_queues[i].front();
                                        batch_queues[i].pop();
                                        batch_counts[i]--;
                                        has_data = true;
                                    }
                                }
                                if (!has_data) {
                                    if (data_loading_done && batch_counts[i] == 0) break;
                                    continue;
                                }

                                // Forward pass
                                auto output = model->as<CustomNet>()->forward(data);
                                auto loss = torch::nn::functional::cross_entropy(output, target);

                                // Backward pass
                                optimizer.zero_grad();
                                loss.backward();

                                // Synchronize gradients
                                synchronize_gradients(i);
                                batches_processed++;
                            }
                            // std::cout << "Training thread " << i << " completed, processed " << batches_processed << " batches" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error in training thread " << i << ": " << e.what() << std::endl;
                        }
                    });
                }

                // Join threads
                data_thread.join();
                for (auto& thread : training_threads) {
                    thread.join();
                }

                // Check for data thread error
                if (data_thread_error) {
                    std::cerr << "Skipping optimizer step due to data thread error" << std::endl;
                    continue; // Skip to next epoch
                }

                // Perform optimizer step
                optimizer.step();
                broadcast_parameters();

                // Synchronize CUDA devices
                torch::cuda::synchronize();

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                // std::cout << "Epoch " << epoch + 1 << " completed in " << duration << " ms\n";
                start_time = end_time;
            } catch (const std::exception& e) {
                std::cerr << "Error in epoch " << epoch + 1 << ": " << e.what() << std::endl;
                continue; // Continue to next epoch
            }
        }
    }

private:
    void initialize() {
        try {
            for (const auto& device : devices_) {
                // std::cout << "Cloning model to device " << device << std::endl;
                auto model = base_model_->clone();
                model->to(device, /*non_blocking=*/true);
                models_.push_back(model);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in initialize: " << e.what() << std::endl;
            throw;
        }
    }

    void synchronize_gradients(size_t device_idx) {
        std::lock_guard<std::mutex> lock(grad_mutex_);
        if (device_idx == 0) {
            try {
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
            } catch (const std::exception& e) {
                std::cerr << "Error in synchronize_gradients: " << e.what() << std::endl;
                throw;
            }
        }
    }

    void broadcast_parameters() {
        try {
            auto params_0 = models_[0]->parameters();
            for (size_t i = 1; i < models_.size(); ++i) {
                auto params_i = models_[i]->parameters();
                for (size_t j = 0; j < params_i.size(); ++j) {
                    auto temp = params_0[j].to(devices_[i], /*non_blocking=*/true).detach().clone();
                    params_i[j].data().copy_(temp);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in broadcast_parameters: " << e.what() << std::endl;
            throw;
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
    try {
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
            // torch::Device(torch::kCUDA, 1),
            torch::Device(torch::kCPU)
        };

        // Create DataParallel (batch size divisible by 2)
        DataParallel dp(model, devices, 128);

        // Create dataset with reordered transforms
        auto dataset = torch::data::datasets::MNIST("/home/kami/Documents/datasets/MNIST/raw/")
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(Flatten())
            .map(torch::data::transforms::Stack<>());

        // Log raw dataset shape for debugging
        auto sample = dataset.get_batch(0);
        std::cout << "Raw MNIST sample shape: " << sample.data.sizes() << std::endl;

        // Create optimizer
        torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));
        auto start = std::chrono::high_resolution_clock::now();
        // Train
        dp.train(dataset, optimizer, 50);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << duration << " s\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Main error: " << e.what() << std::endl;
        return 1;
    }
}