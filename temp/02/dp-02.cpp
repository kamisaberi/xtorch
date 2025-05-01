#include <torch/torch.h>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <memory>
#include <iostream>

// Custom neural network module inheriting from torch::nn::Cloneable
struct CustomNet : torch::nn::Cloneable<CustomNet> {
    CustomNet() {
        // Initialize submodules in reset()
        reset();
    }

    // Implement reset() to initialize submodules
    void reset() override {
        fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(784, 256).bias(true)));
        fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(256, 10).bias(true)));
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    // Submodules
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

// DataParallel class for multi-device training (supports CPU and GPUs)
class DataParallel {
public:
    // Constructor
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
                    // Move data to the first device (e.g., cuda:0)
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

                        // Forward pass using CustomNet
                        auto output = model->as<CustomNet>()->forward(mini_data);
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

            // Synchronize CUDA devices if any are used
            for (const auto& device : devices_) {
                if (device.is_cuda()) {
                    torch::cuda::synchronize();
                    break; // Synchronize once if any CUDA device is used
                }
            }
            std::cout << "Epoch " << epoch + 1 << " completed\n";
        }
    }

private:
    void initialize() {
        // Replicate model to all devices
        for (const auto& device : devices_) {
            // Clone the model
            auto model = base_model_->clone();
            model->to(device);
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
                        auto grad = params_i[j].grad().to(devices_[0]).contiguous();
                        if (params_0[j].grad().defined()) {
                            // Create a new gradient tensor and assign it
                            auto new_grad = params_0[j].grad().clone().to(devices_[0]).contiguous();
                            new_grad += grad; // Out-of-place addition
                            params_0[j].mutable_grad() = new_grad; // Safe assignment
                        } else {
                            // Initialize gradient
                            params_0[j].mutable_grad() = grad.clone();
                        }
                    }
                }
            }
        }
    }

    void broadcast_parameters() {
        // Copy parameters from main model to all other models
        auto params_0 = models_[0]->parameters();
        for (size_t i = 1; i < models_.size(); ++i) {
            auto params_i = models_[i]->parameters();
            for (size_t j = 0; j < params_i.size(); ++j) {
                params_i[j].copy_(params_0[j].to(devices_[i]));
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
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. Exiting." << std::endl;
        return 1;
    }

    // Define custom model
    auto model = std::make_shared<CustomNet>();

    // Define devices (2 GPUs + 1 CPU)
    std::vector<torch::Device> devices = {
        torch::Device(torch::kCUDA, 0), // GPU 0
        torch::Device(torch::kCUDA, 1), // GPU 1
        torch::Device(torch::kCPU)      // CPU
    };

    // Create DataParallel (batch size must be divisible by 3)
    DataParallel dp(model, devices, 60); // Use 60 to ensure divisibility (60 / 3 = 20)

    // Create dataset and dataloader
    auto dataset = torch::data::datasets::MNIST("./data")
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());
    auto dataloader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions().batch_size(60));

    // Create optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));

    // Train
    dp.train(*dataloader, optimizer, 5);

    return 0;
}