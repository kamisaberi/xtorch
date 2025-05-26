#include "include/base/module1.h"
#include <iostream>

namespace xt {

// Derived class to demonstrate overriding the virtual forward method
class MyModule : public Module1 {
public:
    MyModule() {
        // Register module (optional, for PyTorch compatibility)
        register_module("MyModule", nullptr);
    }

    // Override the virtual forward method
    torch::Tensor forward(std::vector<torch::Tensor> tensors) override {
        if (tensors.empty()) {
            throw std::invalid_argument("At least one tensor must be provided");
        }
        if (tensors.size() > 10) {
            throw std::invalid_argument("Maximum 10 tensors allowed");
        }

        // Example implementation: Sum all tensors (assuming compatible shapes)
        torch::Tensor result = tensors[0];
        for (size_t i = 1; i < tensors.size(); ++i) {
            result = result + tensors[i];
        }
        return result;
    }
};

} // namespace xt

// Example usage
int main() {
    // Create an instance of MyModule
    xt::MyModule module;

    // Example tensors
    torch::Tensor t1 = torch::ones({2, 2});
    torch::Tensor t2 = torch::ones({2, 2}) * 2;
    torch::Tensor t3 = torch::ones({2, 2}) * 3;
    torch::Tensor t4 = torch::ones({2, 2}) * 4;

    try {
        // Call variadic forward with 1, 2, 3, and 4 tensors
        torch::Tensor result1 = module.forward(t1);
        torch::Tensor result2 = module.forward(t1, t2);
        torch::Tensor result3 = module.forward(t1, t2, t3);
        torch::Tensor result4 = module.forward(t1, t2, t3, t4);

        // Print results
        std::cout << "Result with 1 tensor:\n" << result1 << std::endl;
        std::cout << "Result with 2 tensors:\n" << result2 << std::endl;
        std::cout << "Result with 3 tensors:\n" << result3 << std::endl;
        std::cout << "Result with 4 tensors:\n" << result4 << std::endl;

        // Test vector-based forward
        std::vector<torch::Tensor> tensors = {t1, t2, t3};
        torch::Tensor result_vec = module.forward(tensors);
        std::cout << "Result with vector-based forward:\n" << result_vec << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}