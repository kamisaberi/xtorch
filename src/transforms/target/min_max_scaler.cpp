#include "include/transforms/target/min_max_scaler.h"

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

void test_scaler(xt::transforms::target::MinMaxScaler& scaler, double value) {
    std::cout << "Input: " << value;
    auto scaled_any = scaler.forward({value});
    auto scaled_value = std::any_cast<double>(scaled_any);
    std::cout << ", Scaled Output: " << scaled_value << std::endl;
}

int main() {
    // 1. --- Setup ---
    // Let's say we analyzed our training data (e.g., house prices) and found
    // the minimum price was $50,000 and the maximum was $450,000.
    double min_val = 50000.0;
    double max_val = 450000.0;

    xt::transforms::target::MinMaxScaler scaler(min_val, max_val);
    std::cout.precision(10);
    std::cout << "Scaler 'fitted' with min=" << min_val << " and max=" << max_val << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing Scaling ---" << std::endl;

    // Case 1: A value at the minimum should become 0.
    test_scaler(scaler, 50000.0);

    // Case 2: A value at the maximum should become 1.
    test_scaler(scaler, 450000.0);

    // Case 3: A value exactly in the middle should become 0.5.
    // Middle = 50k + (450k-50k)/2 = 250k
    test_scaler(scaler, 250000.0);

    // Case 4: A value outside the original range. The scaler will project it
    // beyond the [0, 1] range, which is the correct behavior.
    test_scaler(scaler, 550000.0); // Should be > 1.0

    // --- Test Error Handling ---
    std::cout << "\n--- Testing Error Handling ---" << std::endl;
    try {
        // A scaler where min and max are the same would cause division by zero.
        xt::transforms::target::MinMaxScaler bad_scaler(100.0, 100.0);
    } catch(const std::invalid_argument& e) {
        std::cout << "Caught expected exception for min==max: " << e.what() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    MinMaxScaler::MinMaxScaler(double data_min, double data_max)
        : data_min_(data_min) {

        if (data_max == data_min) {
            throw std::invalid_argument(
                "In MinMaxScaler, data_max cannot be equal to data_min as it would cause division by zero."
            );
        }

        // Pre-calculate the range for efficiency in the forward pass.
        data_range_ = data_max - data_min;
    }

    auto MinMaxScaler::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("MinMaxScaler::forward received an empty list.");
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
             throw std::invalid_argument("Input to MinMaxScaler must be a scalar numeric type.");
        }

        // 2. --- Core Logic: Apply the scaling formula ---
        double scaled_value = (input_value - data_min_) / data_range_;

        return scaled_value;
    }

} // namespace xt::transforms::target