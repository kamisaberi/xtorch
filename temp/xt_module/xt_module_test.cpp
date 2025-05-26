#include "../../include/base/module1.h"
#include <iostream>

#include <iostream>

namespace xt {

// Derived class specializing the forward method
class MyModule : public Module1 {
public:
    MyModule() = default; // No register_module call needed

    // Specialize the variadic forward method
    template<typename... Args>
    torch::Tensor forward(Args... args) {
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

        // Check that all tensors have compatible shapes for concatenation
        auto reference_sizes = tensors[0].sizes();
        for (size_t i = 1; i < tensors.size(); ++i) {
            auto sizes = tensors[i].sizes();
            if (sizes.size() != reference_sizes.size() || sizes[1] != reference_sizes[1]) {
                throw std::invalid_argument("All tensors must have compatible shapes for concatenation");
            }
        }

        // Concatenate tensors along dimension 0 (custom implementation)
        return torch::cat(tensors, /*dim=*/0);
    }
};

// // Explicit template instantiations for MyModule::forward
// extern template torch::Tensor MyModule::forward(torch::Tensor);
// extern template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor);
// extern template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor);
// extern template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// extern template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// extern template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// extern template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// extern template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// extern template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// extern template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

} // namespace xt

// Implementations for MyModule::forward
// namespace xt {
// template torch::Tensor MyModule::forward(torch::Tensor);
// template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor);
// template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor);
// template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// template torch::Tensor MyModule::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
// } // namespace xt

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
        torch::Tensor result1 = module.forward(t1);
        torch::Tensor result2 = module.forward(t1, t2);
        torch::Tensor result3 = module.forward(t1, t2, t3);
        torch::Tensor result4 = module.forward(t1, t2, t3, t4);
        torch::Tensor result5 = module.forward(t1, t2, t3, t4, t5);
        torch::Tensor result6 = module.forward(t1, t2, t3, t4, t5, t6);
        torch::Tensor result7 = module.forward(t1, t2, t3, t4, t5, t6, t7);
        torch::Tensor result8 = module.forward(t1, t2, t3, t4, t5, t6, t7, t8);
        torch::Tensor result9 = module.forward(t1, t2, t3, t4, t5, t6, t7, t8, t9);
        torch::Tensor result10 = module.forward(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10);

        // Print results
        std::cout << "Result with 1 tensor:\n" << result1 << std::endl;
        std::cout << "Result with 2 tensors:\n" << result2 << std::endl;
        std::cout << "Result with 3 tensors:\n" << result3 << std::endl;
        std::cout << "Result with 4 tensors:\n" << result4 << std::endl;
        std::cout << "Result with 5 tensors:\n" << result5 << std::endl;
        std::cout << "Result with 6 tensors:\n" << result6 << std::endl;
        std::cout << "Result with 7 tensors:\n" << result7 << std::endl;
        std::cout << "Result with 8 tensors:\n" << result8 << std::endl;
        std::cout << "Result with 9 tensors:\n" << result9 << std::endl;
        std::cout << "Result with 10 tensors:\n" << result10 << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}