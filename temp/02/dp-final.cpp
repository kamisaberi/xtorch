#include <torch/torch.h>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <memory>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include "../../include/data_parallels/data_parallel.h"

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
            // torch::Device(torch::kCUDA, 0),
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