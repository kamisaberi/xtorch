#include "include/transforms/target/target_encoder.h"

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

void test_encoder(xt::transforms::target::TargetEncoder& encoder, const std::string& city) {
    auto encoded_any = encoder.forward({city});
    auto encoded_value = std::any_cast<double>(encoded_any);
    std::cout << "Encoding for \"" << city << "\": " << encoded_value << std::endl;
}

int main() {
    // 1. --- Setup ---
    // Imagine our target is "house_price" and we're encoding the "city".
    // We have pre-calculated these statistics from our training data:
    double global_avg_price = 350000.0;
    std::unordered_map<std::string, double> city_avg_prices = {
        {"New York", 500000.0},
        {"San Francisco", 750000.0},
        {"Austin", 300000.0}
    };

    xt::transforms::target::TargetEncoder encoder(city_avg_prices, global_avg_price);
    std::cout.precision(10);
    std::cout << "TargetEncoder 'fitted' with pre-calculated means." << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing Target Encoding ---" << std::endl;

    // Case 1: A known category
    test_encoder(encoder, "New York"); // Expected: 500000.0

    // Case 2: Another known category
    test_encoder(encoder, "Austin"); // Expected: 300000.0

    // Case 3: An unseen category. Should fall back to the global mean.
    test_encoder(encoder, "Chicago"); // Expected: 350000.0

    return 0;
}
*/

namespace xt::transforms::target {

    TargetEncoder::TargetEncoder(
            const std::unordered_map<std::string, double>& category_means,
            double global_mean
    ) : category_means_(category_means), global_mean_(global_mean) {}

    auto TargetEncoder::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("TargetEncoder::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        std::string key;

        // Convert the input label to a string key.
        if (input_any.type() == typeid(std::string)) {
            key = std::any_cast<std::string>(input_any);
        } else if (input_any.type() == typeid(const char*)) {
            key = std::any_cast<const char*>(input_any);
        } else if (input_any.type() == typeid(int)) {
            key = std::to_string(std::any_cast<int>(input_any));
        } else if (input_any.type() == typeid(long)) {
            key = std::to_string(std::any_cast<long>(input_any));
        } else {
            throw std::invalid_argument("Input to TargetEncoder must be a string or integer-like type.");
        }

        // 2. --- Core Logic: Lookup the mean ---
        auto it = category_means_.find(key);

        if (it != category_means_.end()) {
            // Category was found, return its pre-calculated mean.
            return it->second;
        } else {
            // Category was not seen during fitting, fall back to the global mean.
            return global_mean_;
        }
    }

} // namespace xt::transforms::target```