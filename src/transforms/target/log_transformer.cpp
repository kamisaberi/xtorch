#include "include/transforms/target/log_transformer.h"

#include <stdexcept>
#include <cmath> // For std::log

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

void test_transformer(xt::transforms::target::LogTransformer& transformer, double value) {
    std::cout << "Input: " << value;
    auto transformed_any = transformer.forward({value});
    auto transformed_value = std::any_cast<double>(transformed_any);
    std::cout << ", log(1 + " << value << ") = " << transformed_value << std::endl;
}

int main() {
    // 1. --- Setup ---
    xt::transforms::target::LogTransformer transformer;
    std::cout.precision(10);

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing Log Transformation ---" << std::endl;

    // Case 1: A standard positive value
    test_transformer(transformer, 99.0); // log(100) = 4.605...

    // Case 2: The special case of zero
    test_transformer(transformer, 0.0); // log(1) = 0.0

    // --- Test Error Handling ---
    std::cout << "\n--- Testing Error Handling ---" << std::endl;
    try {
        // A negative value, where log is undefined.
        test_transformer(transformer, -10.0);
    } catch(const std::invalid_argument& e) {
        std::cout << "Caught expected exception for input -10.0: " << e.what() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    LogTransformer::LogTransformer() = default;

    auto LogTransformer::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("LogTransformer::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        double input_value;

        // Handle various numeric types by casting to double.
        if (input_any.type() == typeid(double)) {
            input_value = std::any_cast<double>(input_any);
        } else if (input_any.type() == typeid(float)) {
            input_value = static_cast<double>(std::any_cast<float>(input_any));
        } else if (input_any.type() == typeid(int)) {
            input_value = static_cast<double>(std::any_cast<int>(input_any));
        } else if (input_any.type() == typeid(long)) {
            input_value = static_cast<double>(std::any_cast<long>(input_any));
        } else {
             throw std::invalid_argument("Input to LogTransformer must be a scalar numeric type.");
        }

        // 2. --- Core Logic ---
        // The logarithm is only defined for positive numbers.
        // We use the log(1+x) transformation, which is standard practice as it
        // gracefully handles x=0 (log(1)=0) and is defined for all x >= 0.
        if (input_value < 0.0) {
            throw std::invalid_argument("Log transformation is not defined for negative inputs.");
        }

        return std::log(1.0 + input_value);
    }

} // namespace xt::transforms::target