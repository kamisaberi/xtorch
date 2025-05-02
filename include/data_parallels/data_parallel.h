#pragma once
#include <torch/torch.h>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <memory>
#include <iostream>
#include <chrono>
#include <stdexcept>

namespace xt::parallelism
{
    // Custom transform to flatten MNIST images
    struct Flatten : public torch::data::transforms::TensorTransform<torch::Tensor>
    {
        torch::Tensor operator()(torch::Tensor input)
        {
            // Log input shape for debugging
            // std::cout << "Flatten input shape: " << input.sizes() << std::endl;
            // Input: [1, 28, 28] (single sample) -> Output: [784]
            if (input.dim() >= 3)
            {
                return input.view({-1});
            }
            return input;
        }
    };


    // Custom neural network module inheriting from torch::nn::Cloneable
    struct CustomNet : torch::nn::Cloneable<CustomNet>
    {
        CustomNet()
        {
            reset();
        }

        void reset() override
        {
            fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(784, 256).bias(true)));
            fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(256, 10).bias(true)));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            x = torch::relu(fc1->forward(x));
            x = fc2->forward(x);
            return x;
        }

        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    };


    // Optimized DataParallel class for multi-GPU training
    class DataParallel
    {
    public:
        DataParallel(std::shared_ptr<torch::nn::Module> model, const std::vector<torch::Device>& devices,
                     size_t batch_size);

        template <typename DatasetType>
        void train(DatasetType& dataset, torch::optim::Optimizer& optimizer, size_t epochs)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            for (size_t epoch = 0; epoch < epochs; ++epoch)
            {
                try
                {
                    // Create fresh DataLoader for each epoch
                    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                        dataset, torch::data::DataLoaderOptions().batch_size(batch_size_).workers(4));

                    std::vector<std::queue<std::tuple<torch::Tensor, torch::Tensor>>> batch_queues(devices_.size());
                    std::vector<std::mutex> queue_mutexes(devices_.size());
                    std::vector<size_t> batch_counts(devices_.size(), 0);
                    std::atomic<bool> data_loading_done{false};
                    bool data_thread_error = false;

                    // Single data distribution thread
                    std::thread data_thread([&]()
                    {
                        try
                        {
                            size_t device_idx = 0;
                            // std::cout << "Data thread started for epoch " << epoch + 1 << std::endl;
                            for (auto& batch : *dataloader)
                            {
                                auto data = batch.data;
                                auto target = batch.target;

                                // Log shape after stacking
                                // std::cout << "Batch data shape after stacking: " << data.sizes() << std::endl;

                                // Verify data shape
                                if (data.dim() != 2 || data.size(1) != 784)
                                {
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
                        }
                        catch (const std::exception& e)
                        {
                            std::cerr << "Error in data thread: " << e.what() << std::endl;
                            data_loading_done = true;
                            data_thread_error = true;
                        }
                    });

                    // Training threads for each device
                    std::vector<std::thread> training_threads;
                    for (size_t i = 0; i < devices_.size(); ++i)
                    {
                        training_threads.emplace_back([&, i]()
                        {
                            try
                            {
                                auto model = models_[i];
                                model->train();
                                // std::cout << "Training thread " << i << " started on " << devices_[i] << std::endl;
                                size_t batches_processed = 0;
                                while (true)
                                {
                                    torch::Tensor data, target;
                                    bool has_data = false;
                                    {
                                        std::lock_guard<std::mutex> lock(queue_mutexes[i]);
                                        if (!batch_queues[i].empty())
                                        {
                                            std::tie(data, target) = batch_queues[i].front();
                                            batch_queues[i].pop();
                                            batch_counts[i]--;
                                            has_data = true;
                                        }
                                    }
                                    if (!has_data)
                                    {
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
                            }
                            catch (const std::exception& e)
                            {
                                std::cerr << "Error in training thread " << i << ": " << e.what() << std::endl;
                            }
                        });
                    }

                    // Join threads
                    data_thread.join();
                    for (auto& thread : training_threads)
                    {
                        thread.join();
                    }

                    // Check for data thread error
                    if (data_thread_error)
                    {
                        std::cerr << "Skipping optimizer step due to data thread error" << std::endl;
                        continue; // Skip to next epoch
                    }

                    // Perform optimizer step
                    optimizer.step();
                    broadcast_parameters();

                    // Synchronize CUDA devices
                    torch::cuda::synchronize();

                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).
                        count();
                    // std::cout << "Epoch " << epoch + 1 << " completed in " << duration << " ms\n";
                    start_time = end_time;
                }
                catch (const std::exception& e)
                {
                    std::cerr << "Error in epoch " << epoch + 1 << ": " << e.what() << std::endl;
                    continue; // Continue to next epoch
                }
            }
        }

    private:
        void initialize();

        void synchronize_gradients(size_t device_idx);

        void broadcast_parameters();

        std::shared_ptr<torch::nn::Module> base_model_;
        std::vector<torch::Device> devices_;
        std::vector<std::shared_ptr<torch::nn::Module>> models_;
        size_t batch_size_;
        std::mutex grad_mutex_;
    };
}
