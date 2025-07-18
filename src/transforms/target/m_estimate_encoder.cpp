#include <transforms/target/m_estimate_encoder.h>

// #include "include/transforms/target/m_estimate_encoder.h"
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

void test_encoder(xt::transforms::target::MEstimateEncoder& encoder, const std::string& country) {
    auto encoded_any = encoder.forward({country});
    auto encoded_value = std::any_cast<double>(encoded_any);
    std::cout << "Encoded value for \"" << country << "\": " << encoded_value << std::endl;
}

int main() {
    // 1. --- Setup: Pre-calculated Statistics ---
    // Imagine our target is "customer_rating" (1-5) and we're encoding "product_category".
    // Global average rating is 3.5.
    double global_rating_mean = 3.5;

    std::unordered_map<std::string, xt::transforms::target::TargetStats> category_stats = {
        // "Electronics" is very common, its mean should be trusted.
        {"Electronics", {4.5, 1000}}, // Mean: 4.5, Count: 1000
        // "Books" is moderately common.
        {"Books", {4.0, 50}}, // Mean: 4.0, Count: 50
        // "Home Goods" is rare, its mean should be shrunk towards the global mean.
        {"Home Goods", {2.5, 3}} // Mean: 2.5, Count: 3
    };

    // Create the encoder with a smoothing factor `m` of 20.
    // This means a category needs about 20 samples before its own mean gets heavy weight.
    xt::transforms::target::MEstimateEncoder encoder(category_stats, global_rating_mean, 20.0);
    std::cout.precision(10);
    std::cout << "Global Mean Rating: " << global_rating_mean << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing M-Estimate Encoding ---" << std::endl;

    // Case 1: High-count category. Should be very close to its own mean (4.5).
    test_encoder(encoder, "Electronics");

    // Case 2: Medium-count category. Should be shrunk towards 3.5.
    test_encoder(encoder, "Books");

    // Case 3: Low-count category. Should be shrunk heavily towards 3.5.
    test_encoder(encoder, "Home Goods");

    // Case 4: Unseen category. Should fall back to the global mean (3.5).
    test_encoder(encoder, "Gardening");

    return 0;
}
*/

namespace xt::transforms::target {

    MEstimateEncoder::MEstimateEncoder(
        const std::unordered_map<std::string, TargetStatsM>& category_stats,
        double global_mean,
        double m
    ) : category_stats_(category_stats), global_mean_(global_mean), m_(m) {

        if (m_ < 0.0) {
            throw std::invalid_argument("Smoothing factor m cannot be negative.");
        }
    }

    auto MEstimateEncoder::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("MEstimateEncoder::forward received an empty list.");
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
             throw std::invalid_argument("Input to MEstimateEncoder must be a string or integer-like type.");
        }

        // 2. --- Core Logic ---
        auto it = category_stats_.find(key);

        if (it != category_stats_.end()) {
            // --- Category was seen during fitting ---
            const auto& stats = it->second;
            double category_mean = stats.mean;
            double category_count = static_cast<double>(stats.count);

            // M-Estimate formula
            double numerator = (category_count * category_mean) + (m_ * global_mean_);
            double denominator = category_count + m_;

            return numerator / denominator;
        } else {
            // --- Category is unseen ---
            // For unseen categories, the count is 0, so the formula simplifies
            // to (m * global_mean) / m = global_mean.
            return global_mean_;
        }
    }

} // namespace xt::transforms::target