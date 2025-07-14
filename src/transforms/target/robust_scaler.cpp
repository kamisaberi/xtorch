#include "include/transforms/target/robust_scaler.h"

#include <stdexcept>

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

void test_scaler(xt::transforms::target::RobustScaler& scaler, double value) {
    std::cout << "Input: " << value;
    auto scaled_any = scaler.forward({value});
    auto scaled_value = std::any_cast<double>(scaled_any);
    std::cout << ", Scaled Output: " << scaled_value << std::endl;
}

int main() {
    // 1. --- Setup ---
    // Let's say we analyzed our training data (e.g., employee salaries) and found:
    // - 25th percentile (q1): $60,000
    // - Median (50th percentile): $85,000
    // - 75th percentile (q3): $140,000
    double q1 = 60000.0;
    double median = 85000.0;
    double q3 = 140000.0;

    xt::transforms::target::RobustScaler scaler(median, q1, q3);
    std::cout.precision(10);
    std::cout << "Scaler 'fitted' with median=" << median << ", q1=" << q1 << ", q3=" << q3 << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing Scaling ---" << std::endl;

    // Case 1: A value at the median should become 0.
    test_scaler(scaler, 85000.0);

    // Case 2: A value at the 3rd quartile.
    // Result should be (140k - 85k) / (140k - 60k) = 55k / 80k = 0.6875
    test_scaler(scaler, 140000.0);

    // Case 3: A value at the 1st quartile.
    // Result should be (60k - 85k) / (140k - 60k) = -25k / 80k = -0.3125
    test_scaler(scaler, 60000.0);

    // Case 4: An outlier value. It gets scaled but doesn't dominate the range.
    test_scaler(scaler, 500000.0);

    // --- Test Error Handling ---
    std::cout << "\n--- Testing Error Handling ---" << std::endl;
    try {
        // A scaler where q1 and q3 are the same would cause division by zero.
        xt::transforms::target::RobustScaler bad_scaler(100.0, 100.0, 100.0);
    } catch(const std::invalid_argument& e) {
        std::cout << "Caught expected exception for q1==q3: " << e.what() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    RobustScaler::RobustScaler(double median, double q1, double q3)
            : median_(median) {

        if (q3 < q1) {
            throw std::invalid_argument("In RobustScaler, q3 cannot be less than q1.");
        }

        interquartile_range_ = q3 - q1;

        if (interquartile_range_ == 0.0) {
            throw std::invalid_argument(
                    "In RobustScaler, the interquartile range (q3 - q1) cannot be zero, as it would cause division by zero."
            );
        }
    }

    auto RobustScaler::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RobustScaler::forward received an empty list.");
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
            throw std::invalid_argument("Input to RobustScaler must be a scalar numeric type.");
        }

        // 2. --- Core Logic: Apply the scaling formula ---
        double scaled_value = (input_value - median_) / interquartile_range_;

        return scaled_value;
    }

} // namespace xt::transforms::target