#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory> // For std::shared_ptr
#include <thread> // For std::thread
#include <any>    // For std::any

#include "include/base/base.h" // Assuming xt::Module is here

namespace xt {

    // Forward declaration if your actual model type is different
    // using ModelType = xt::Module; // Or your specific model base

    class CustomDataParallel : public xt::Module { // Inherit from your base module
    public:
        // Constructor takes the model to be parallelized and a list of GPU device indices
        CustomDataParallel(std::shared_ptr<xt::Module> module_to_parallelize,
                           const std::vector<torch::Device>& devices);

        ~CustomDataParallel() override;

        // The forward method that will be called by the Trainer
        // It expects a single std::any containing the input data tensor(s)
        // and potentially another for targets if loss is computed internally.
        // For simplicity, let's assume inputs = {data_tensor, target_tensor}
        std::any forward(std::initializer_list<std::any> inputs) override;

        // Method to get the parameters of the primary model replica (for the optimizer)
        std::vector<torch::Tensor> parameters(bool recurse = true) const override;
        std::vector<torch::Tensor> named_parameters(bool recurse = true) const override; // If your xt::Module has it

        // Method to synchronize parameters from the primary replica to all other replicas
        void synchronize_replicas();

        // Overload to() to move all replicas and the original module
        void to(torch::Device device, bool non_blocking = false) override;
        void to(torch::ScalarType dtype, bool non_blocking = false) override;
        void to(torch::Device device, torch::ScalarType dtype, bool non_blocking = false) override;


    private:
        void scatter(const torch::Tensor& input_tensor, std::vector<torch::Tensor>& scattered_inputs);
        torch::Tensor gather(const std::vector<torch::Tensor>& scattered_outputs, int64_t target_batch_dim);

        std::shared_ptr<xt::Module> original_module_; // The module before replication
        std::vector<torch::Device> devices_;
        torch::Device primary_device_; // Usually devices_[0]

        std::vector<std::shared_ptr<xt::Module>> replicas_; // Model replicas on each device

        // For thread management
        std::vector<std::thread> threads_;
        // You might need mutexes/condition variables if using more complex shared state,
        // but for this scatter/gather, direct results can be used.
    };

} // namespace xt