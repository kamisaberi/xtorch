#include "include/transforms/target/power_transformer.h"

#include <stdexcept>
#include <cmath> // For std::log, std::pow

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

void test_transformer(xt::transforms::target::PowerTransformer& transformer, double value) {
    std::cout << "Input: " << value;
    try {
        auto transformed_any = transformer.forward({value});
        auto transformed_value = std::any_cast<double>(transformed_any);
        std::cout << ", Transformed Output: " << transformed_value << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << ", Error: " << e.what() << std::endl;
    }
}

int main() {
    std::cout.precision(10);

    // --- Example 1: Yeo-Johnson (lambda = 0.5) ---
    std::cout << "--- Testing Yeo-Johnson (lambda=0.5) ---" << std::endl;
    xt::transforms::target::PowerTransformer yj_transformer(0.5);
    test_transformer(yj_transformer, 10.0);  // Positive value
    test_transformer(yj_transformer, -10.0); // Negative value

    // --- Example 2: Yeo-Johnson (lambda = 0, behaves like log) ---
    std::cout << "\n--- Testing Yeo-Johnson (lambda=0) ---" << std::endl;
    xt::transforms::target::PowerTransformer yj_log_transformer(0.0);
    test_transformer(yj_log_transformer, 10.0); // log(11) = 2.397...

    // --- Example 3: Box-Cox (lambda = 0.5) ---
    std::cout << "\n--- Testing Box-Cox (lambda=0.5) ---" << std::endl;
    xt::transforms::target::PowerTransformer bc_transformer(0.5, xt::transforms::target::PowerTransformType::BOX_COX);
    test_transformer(bc_transformer, 10.0); // Positive value
    test_transformer(bc_transformer, -10.0);// Negative value, should error

    // --- Example 4: Box-Cox (lambda = 0, behaves like log) ---
    std::cout << "\n--- Testing Box-Cox (lambda=0) ---" << std::endl;
    xt::transforms::target::PowerTransformer bc_log_transformer(0.0, xt::transforms::target::PowerTransformType::BOX_COX);
    test_transformer(bc_log_transformer, 10.0); // log(10) = 2.302...

    return 0;
}
*/

namespace xt::transforms::target {

    PowerTransformer::PowerTransformer(double lambda, PowerTransformType type)
            : lambda_(lambda), type_(type) {}

    auto PowerTransformer::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("PowerTransformer::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        double x;

        if (input_any.type() == typeid(double)) {
            x = std::any_cast<double>(input_any);
        } else if (input_any.type() == typeid(float)) {
            x = static_cast<double>(std::any_cast<float>(input_any));
        } else if (input_any.type() == typeid(int)) {
            x = static_cast<double>(std::any_cast<int>(input_any));
        } else if (input_any.type() == typeid(long)) {
            x = static_cast<double>(std::any_cast<long>(input_any));
        } else {
            throw std::invalid_argument("Input to PowerTransformer must be a scalar numeric type.");
        }

        // 2. --- Apply selected transformation ---
        if (type_ == PowerTransformType::YEO_JOHNSON) {
            if (x >= 0) {
                if (lambda_ == 0.0) {
                    return std::log(x + 1.0);
                } else {
                    return (std::pow(x + 1.0, lambda_) - 1.0) / lambda_;
                }
            } else { // x < 0
                if (lambda_ == 2.0) {
                    return -std::log(-x + 1.0);
                } else {
                    return -(std::pow(-x + 1.0, 2.0 - lambda_) - 1.0) / (2.0 - lambda_);
                }
            }
        } else { // BOX_COX
            if (x <= 0) {
                throw std::invalid_argument("Box-Cox transformation requires strictly positive inputs.");
            }
            if (lambda_ == 0.0) {
                return std::log(x);
            } else {
                return (std::pow(x, lambda_) - 1.0) / lambda_;
            }
        }
    }

} // namespace xt::transforms::target