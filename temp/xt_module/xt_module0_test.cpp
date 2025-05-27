#include "include/base/module1.h"
#include <iostream>
#include <vector>

namespace xt {

// Derived class overriding forward
class MyModule : public Module1 {
public:
    MyModule() = default; // No register_module

    // Override forward to take exactly 2 tensors and return std::vector<float>
    auto forward(std::initializer_list<torch::Tensor> tensors) -> std::any override {
        // Ensure exactly 2 tensors
        if (tensors.size() != 2) {
            throw std::invalid_argument("MyModule::forward requires exactly 2 tensors");
        }

        // Convert initializer_list to vector for processing
        std::vector<torch::Tensor> tensor_vec(tensors);

        // Validate tensors
        for (const auto& tensor : tensor_vec) {
            if (tensor.numel() == 0) {
                throw std::invalid_argument("Tensor is empty");
            }
        }

        // Return first element of each tensor
        std::vector<float> result = {
            tensor_vec[0].flatten()[0].item<float>(),
            tensor_vec[1].flatten()[0].item<float>()
        };
        return std::any(result);
    }

    // Convenience method for two tensors
    auto forward(torch::Tensor t1, torch::Tensor t2) -> std::vector<float> {
        // Call the virtual forward with initializer_list
        auto any_result = forward({t1, t2});
        return std::any_cast<std::vector<float>>(any_result);
    }
};

} // namespace xt

// Test usage
int main() {
    // Create MyModule instance
    xt::MyModule module;

    // Example tensors (2x2)
    torch::Tensor t1 = torch::ones({2, 2}) * 1;
    torch::Tensor t2 = torch::ones({2, 2}) * 2;
    torch::Tensor t3 = torch::ones({2, 2}) * 3;

    try {
        // Test operator() with 2 tensors
        auto result_op = module({t1, t2});
        auto result_vec = std::any_cast<std::vector<float>>(result_op);
        std::cout << "operator() result with 2 tensors: ";
        for (float val : result_vec) std::cout << val << " ";
        std::cout << "\n";

        // Test MyModule::forward with 2 tensors
        auto result_forward = module.forward(t1, t2);
        std::cout << "forward result with 2 tensors: ";
        for (float val : result_forward) std::cout << val << " ";
        std::cout << "\n";

        // Test error with wrong number of tensors
        try {
            module({t1, t2, t3}); // Should fail
        } catch (const std::exception& e) {
            std::cout << "Expected error for 3 tensors: " << e.what() << "\n";
        }

        // Test error with empty tensor
        try {
            torch::Tensor t_empty = torch::Tensor();
            module({t_empty, t1}); // Should fail
        } catch (const std::exception& e) {
            std::cout << "Expected error for empty tensor: " << e.what() << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "Unknown error occurred\n";
    }

    return 0;
}