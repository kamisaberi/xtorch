#include "include/base/module1.h"
#include <iostream>
#include <vector>

namespace xt {

// Derived class overriding forward
class MyModule : public Module1 {
public:
    MyModule() = default;

    auto forward(std::initializer_list<std::any> data) -> std::any override {
        if (data.size() != 2) {
            throw std::invalid_argument("MyModule::forward requires exactly 2 tensors");
        }
        std::vector<torch::Tensor> tensors;
        try {
            for (const auto& item : data) {
                tensors.push_back(std::any_cast<torch::Tensor>(item));
            }
        } catch (const std::bad_any_cast&) {
            throw std::invalid_argument("MyModule::forward expects torch::Tensor elements");
        }
        for (const auto& tensor : tensors) {
            if (tensor.numel() == 0) {
                throw std::invalid_argument("Tensor is empty");
            }
        }
        std::vector<float> result = {
                tensors[0].flatten()[0].item<float>(),
                tensors[1].flatten()[0].item<float>()
        };
        return std::any(result);
    }

    auto forward(torch::Tensor t1, torch::Tensor t2) -> std::vector<float> {
        auto any_result = forward({std::any(t1), std::any(t2)});
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