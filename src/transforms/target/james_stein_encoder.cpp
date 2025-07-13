#include "include/transforms/target/james_stein_encoder.h"

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

void test_encoder(xt::transforms::target::JamesSteinEncoder& encoder, const std::string& country) {
    auto encoded_any = encoder.forward({country});
    auto encoded_value = std::any_cast<double>(encoded_any);
    std::cout << "Encoded value for \"" << country << "\": " << encoded_value << std::endl;
}

int main() {
    // 1. --- Setup: Pre-calculated Statistics ---
    // Imagine our target is "income" and we're encoding "country".
    // Global average income across all countries is $50,000.
    double global_income_mean = 50000.0;

    std::unordered_map<std::string, xt::transforms::target::CategoryStats> country_stats = {
        // "USA" has many samples, so its mean should be trusted.
        {"USA", {70000.0, 1000}}, // Mean: 70k, Count: 1000
        // "Canada" has a moderate number of samples.
        {"Canada", {60000.0, 100}}, // Mean: 60k, Count: 100
        // "Andorra" is very rare, so its mean should be heavily shrunk towards the global mean.
        {"Andorra", {90000.0, 3}} // Mean: 90k, Count: 3
    };

    // Create the encoder with a smoothing factor of 20.
    xt::transforms::target::JamesSteinEncoder encoder(country_stats, global_income_mean, 20.0);
    std::cout.precision(10); // Print with more precision to see the effect.
    std::cout << "Global Mean: " << global_income_mean << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing James-Stein Encoding ---" << std::endl;

    // Case 1: High-count category. Should be very close to its own mean (70k).
    test_encoder(encoder, "USA");

    // Case 2: Medium-count category. Should be shrunk slightly towards 50k.
    test_encoder(encoder, "Canada");

    // Case 3: Low-count category. Should be shrunk heavily towards 50k.
    test_encoder(encoder, "Andorra");

    // Case 4: Unseen category. Should fall back to the global mean (50k).
    test_encoder(encoder, "France");

    return 0;
}
*/

namespace xt::transforms::target {

    JamesSteinEncoder::JamesSteinEncoder(
        const std::unordered_map<std::string, CategoryStats>& category_stats,
        double global_mean,
        double smoothing
    ) : category_stats_(category_stats), global_mean_(global_mean), smoothing_(smoothing) {

        if (smoothing_ < 0.0) {
            throw std::invalid_argument("Smoothing factor cannot be negative.");
        }
    }

    auto JamesSteinEncoder::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("JamesSteinEncoder::forward received an empty list.");
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
             throw std::invalid_argument("Input to JamesSteinEncoder must be a string or integer-like type.");
        }

        // 2. --- Core Logic ---
        auto it = category_stats_.find(key);

        if (it != category_stats_.end()) {
            // --- Category was seen during fitting ---
            const auto& stats = it->second;
            double category_mean = stats.mean;
            double category_count = static_cast<double>(stats.count);

            // Calculate the shrinkage factor B
            double shrinkage = smoothing_ / (smoothing_ + category_count);

            // Calculate the smoothed (shrunk) value
            double encoded_value = (1.0 - shrinkage) * category_mean + shrinkage * global_mean_;

            return encoded_value;
        } else {
            // --- Category is unseen ---
            // Fall back to the global mean, which is the safest estimate.
            return global_mean_;
        }
    }

} // namespace xt::transforms::target