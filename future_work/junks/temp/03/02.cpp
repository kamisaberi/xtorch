#include "base/cloneable.h"
#include "base/module.h"
#include <iostream>



namespace xt {
    // Define an extended module that inherits from xt::Cloneable
    class ExtendedModule : public xt::Cloneable<ExtendedModule> {
    public:
        // Constructor
        ExtendedModule(int64_t input_size, int64_t hidden_size, int64_t output_size);

        // Copy constructor for cloning
        ExtendedModule(const ExtendedModule& other);

        // Forward pass
        torch::Tensor forward(torch::Tensor x) const override;

        // Operator() to allow function-like calls
        torch::Tensor operator()(torch::Tensor x) {
            return forward(x);
        }

        // Reset method to initialize or reinitialize the module
        void reset() override;

    private:
        torch::nn::Linear linear1{nullptr}; // First linear layer submodule
        torch::nn::ReLU relu{nullptr};     // ReLU activation submodule
        torch::nn::Linear linear2{nullptr}; // Second linear layer submodule
    };
}


namespace xt {
    ExtendedModule::ExtendedModule(int64_t input_size, int64_t hidden_size, int64_t output_size) {
        // Initialize the first linear layer and register it as a submodule
        linear1 = register_module("linear1", torch::nn::Linear(input_size, hidden_size));

        // Initialize ReLU activation
        relu = register_module("relu", torch::nn::ReLU());

        // Initialize the second linear layer and register it as a submodule
        linear2 = register_module("linear2", torch::nn::Linear(hidden_size, output_size));

        // Call reset to initialize the module
        reset();
    }

    ExtendedModule::ExtendedModule(const ExtendedModule& other) {
        // Initialize submodules with the same configuration
        linear1 = register_module("linear1", torch::nn::Linear(other.linear1->options));
        relu = register_module("relu", torch::nn::ReLU());
        linear2 = register_module("linear2", torch::nn::Linear(other.linear2->options));

        // Copy parameters and buffers will be handled by clone
        reset();
    }

    torch::Tensor ExtendedModule::forward(torch::Tensor x) const {
        // Apply the first linear layer
        x = linear1->forward(x);

        // Apply ReLU activation
        x = relu->forward(x);
        // Apply the second linear layer
        x = linear2->forward(x);
        return x;
    }

    void ExtendedModule::reset() {
        // Submodules are already initialized in the constructor
    }
}


int main() {
    try {
        // Create an instance of ExtendedModule
        xt::ExtendedModule module(10, 20, 5);

        // Create a sample input tensor
        auto input = torch::randn({2, 10});

        // Perform a forward pass using operator()
        auto output = module(input);
        std::cout << "Output:\n" << output << std::endl;

        // Clone the module
        auto cloned_module = module.clone();
        auto cloned_output = cloned_module->forward(input);
        std::cout << "Cloned Output:\n" << cloned_output << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}