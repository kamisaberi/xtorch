#include <torch/torch.h>
#include <iostream>

// Define a custom module that inherits from torch::nn::Cloneable
class CustomModule : public torch::nn::Cloneable<CustomModule>
{
public:
    CustomModule(int64_t input_size, int64_t output_size);

    torch::Tensor forward(torch::Tensor x);
    torch::Tensor operator()(torch::Tensor x)
    {
        return forward(x);
    }
    void reset() override;

protected:
    torch::nn::Linear linear{nullptr}; // Linear layer submodule
};

CustomModule::CustomModule(int64_t input_size, int64_t output_size)
{
    linear = register_module("linear", torch::nn::Linear(input_size, output_size));

    reset();
}

torch::Tensor CustomModule::forward(torch::Tensor x)
{
    return linear->forward(x);
}

void CustomModule::reset()
{
}

class ExtendedModule : public torch::nn::Cloneable<ExtendedModule>
{
public:
    ExtendedModule(int64_t input_size, int64_t hidden_size, int64_t output_size);

    torch::Tensor forward(torch::Tensor x);

    torch::Tensor operator()(torch::Tensor x)
    {
        return forward(x);
    }

    void reset() override;

private:
    torch::nn::Linear linear1{nullptr}; // First linear layer submodule
    torch::nn::ReLU relu{nullptr}; // ReLU activation submodule
    torch::nn::Linear linear2{nullptr}; // Second linear layer submodule
};


ExtendedModule::ExtendedModule(int64_t input_size, int64_t hidden_size, int64_t output_size)
{
    // Initialize the first linear layer and register it as a submodule
    linear1 = register_module("linear1", torch::nn::Linear(input_size, hidden_size));

    // Initialize ReLU activation
    relu = register_module("relu", torch::nn::ReLU());

    // Initialize the second linear layer and register it as a submodule
    linear2 = register_module("linear2", torch::nn::Linear(hidden_size, output_size));

    // Call reset to initialize the module
    reset();
}

torch::Tensor ExtendedModule::forward(torch::Tensor x)
{
    // Apply the first linear layer
    x = linear1->forward(x);
    // Apply ReLU activation
    x = relu->forward(x);
    // Apply the second linear layer
    x = linear2->forward(x);
    return x;
}

void ExtendedModule::reset()
{
    // Submodules are already initialized in the constructor
}


int main()
{
    try
    {
        // Create an instance of ExtendedModule
        ExtendedModule module(10, 20, 5);

        // Create a sample input tensor
        auto input = torch::randn({2, 10});

        // Perform a forward pass using operator()
        auto output = module(input);
        std::cout << "Output:\n" << output << std::endl;

        // Clone the module
        auto cloned_module = module.clone();
        auto cloned_output = cloned_module->forward(input);
        std::cout << "Cloned Output:\n" << cloned_output << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
