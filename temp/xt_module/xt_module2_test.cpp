#include "../../include/base/module2.h"
#include <iostream>
#include <vector>

namespace xt {

// Derived class specializing the forward method
class MyModule : public Module2 {
public:
    MyModule() = default; // No register_module call needed

    // Specialize the variadic forward method to return std::vector<float>
    template<typename... Args>
    auto forward(Args... args) -> std::vector<float> {
        // Ensure all arguments are torch::Tensor
        static_assert(
            (std::is_same_v<std::decay_t<Args>, torch::Tensor> && ...),
            "All arguments to forward must be torch::Tensor"
        );

        // Collect tensors into a vector
        std::vector<torch::Tensor> tensors = {args...};

        // Check for valid number of arguments (1 to 10)
        if (tensors.empty()) {
            throw std::invalid_argument("At least one tensor must be provided");
        }
        if (tensors.size() > 10) {
            throw std::invalid_argument("Maximum 10 tensors allowed");
        }

        // Sum the first element of each tensor
        std::vector<float> result;
        for (const auto& tensor : tensors) {
            // Ensure tensor is not empty and access first element
            if (tensor.numel() == 0) {
                throw std::invalid_argument("Tensor is empty");
            }
            result.push_back(tensor.flatten()[0].item<float>());
        }

        return result;
    }
};

} // namespace xt

// Example usage
int main() {
    // Create an instance of MyModule
    xt::MyModule module;

    // Example tensors (all 2x2 for simplicity)
    torch::Tensor t1 = torch::ones({2, 2}) * 1;
    torch::Tensor t2 = torch::ones({2, 2}) * 2;
    torch::Tensor t3 = torch::ones({2, 2}) * 3;
    torch::Tensor t4 = torch::ones({2, 2}) * 4;
    torch::Tensor t5 = torch::ones({2, 2}) * 5;
    torch::Tensor t6 = torch::ones({2, 2}) * 6;
    torch::Tensor t7 = torch::ones({2, 2}) * 7;
    torch::Tensor t8 = torch::ones({2, 2}) * 8;
    torch::Tensor t9 = torch::ones({2, 2}) * 9;
    torch::Tensor t10 = torch::ones({2, 2}) * 10;

    try {
        // Test variadic forward with 1 to 10 tensors
        auto result1 = module.forward(t1);
        auto result2 = module.forward(t1, t2);
        auto result3 = module.forward(t1, t2, t3);
        auto result4 = module.forward(t1, t2, t3, t4);
        auto result5 = module.forward(t1, t2, t3, t4, t5);
        auto result6 = module.forward(t1, t2, t3, t4, t5, t6);
        auto result7 = module.forward(t1, t2, t3, t4, t5, t6, t7);
        auto result8 = module.forward(t1, t2, t3, t4, t5, t6, t7, t8);
        auto result9 = module.forward(t1, t2, t3, t4, t5, t6, t7, t8, t9);
        auto result10 = module.forward(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10);

        // Print results
        std::cout << "Result with 1 tensor: ";
        for (float val : result1) std::cout << val << " ";
        std::cout << "\n";

        std::cout << "Result with 2 tensors: ";
        for (float val : result2) std::cout << val << " ";
        std::cout << "\n";

        std::cout << "Result with 3 tensors: ";
        for (float val : result3) std::cout << val << " ";
        std::cout << "\n";

        std::cout << "Result with 4 tensors: ";
        for (float val : result4) std::cout << val << " ";
        std::cout << "\n";

        std::cout << "Result with 5 tensors: ";
        for (float val : result5) std::cout << val << " ";
        std::cout << "\n";

        std::cout << "Result with 6 tensors: ";
        for (float val : result6) std::cout << val << " ";
        std::cout << "\n";

        std::cout << "Result with 7 tensors: ";
        for (float val : result7) std::cout << val << " ";
        std::cout << "\n";

        std::cout << "Result with 8 tensors: ";
        for (float val : result8) std::cout << val << " ";
        std::cout << "\n";

        std::cout << "Result with 9 tensors: ";
        for (float val : result9) std::cout << val << " ";
        std::cout << "\n";

        std::cout << "Result with 10 tensors: ";
        for (float val : result10) std::cout << val << " ";
        std::cout << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}