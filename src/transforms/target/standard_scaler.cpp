#include <transforms/target/standard_scaler.h>

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

void test_scaler(xt::transforms::target::StandardScaler& scaler, double value) {
    std::cout << "Input: " << value;
    auto scaled_any = scaler.forward({value});
    auto scaled_value = std::any_cast<double>(scaled_any);
    std::cout << ", Scaled Output (z-score): " << scaled_value << std::endl;
}

int main() {
    // 1. --- Setup ---
    // Let's say we analyzed our training data (e.g., test scores) and found:
    // - Mean score: 80.0
    // - Standard deviation: 10.0
    double mean = 80.0;
    double std_dev = 10.0;

    xt::transforms::target::StandardScaler scaler(mean, std_dev);
    std::cout.precision(10);
    std::cout << "Scaler 'fitted' with mean=" << mean << " and std_dev=" << std_dev << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing Scaling ---" << std::endl;

    // Case 1: A value at the mean should become 0.
    test_scaler(scaler, 80.0);

    // Case 2: A value one standard deviation above the mean should become 1.
    test_scaler(scaler, 90.0);

    // Case 3: A value two standard deviations below the mean should become -2.
    test_scaler(scaler, 60.0);

    // Case 4: A value in between.
    test_scaler(scaler, 85.0); // (85 - 80) / 10 = 0.5

    // --- Test Error Handling ---
    std::cout << "\n--- Testing Error Handling ---" << std::endl;
    try {
        // A scaler where std_dev is zero would cause division by zero.
        xt::transforms::target::StandardScaler bad_scaler(100.0, 0.0);
    } catch(const std::invalid_argument& e) {
        std::cout << "Caught expected exception for std_dev==0: " << e.what() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    StandardScaler::StandardScaler(double mean, double std_dev)
            : mean_(mean), std_dev_(std_dev) {

        if (std_dev_ == 0.0) {
            throw std::invalid_argument(
                    "In StandardScaler, standard deviation cannot be zero as it would cause division by zero."
            );
        }
    }

    auto StandardScaler::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("StandardScaler::forward received an empty list.");
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
            throw std::invalid_argument("Input to StandardScaler must be a scalar numeric type.");
        }

        // 2. --- Core Logic: Apply the standardization formula ---
        double scaled_value = (input_value - mean_) / std_dev_;

        return scaled_value;
    }

} // namespace xt::transforms::target