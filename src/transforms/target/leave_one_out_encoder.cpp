#include <transforms/target/leave_one_out_encoder.h>

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

void test_encoder(xt::transforms::target::LeaveOneOutEncoder& encoder, const std::string& city, double house_price) {
    auto encoded_any = encoder.forward({city, house_price});
    auto encoded_value = std::any_cast<double>(encoded_any);
    std::cout << "Encoding for a house in \"" << city << "\" with price " << house_price
              << ". Encoded value: " << encoded_value << std::endl;
}

int main() {
    // 1. --- Setup: Pre-calculated Statistics ---
    // Imagine our target is "house_price" and we're encoding the "city".
    // Global average house price is $350,000.
    double global_price_mean = 350000.0;

    // Stats calculated from a training set:
    std::unordered_map<std::string, xt::transforms::target::TargetStats> city_stats = {
        // New York has 3 samples, with a total price sum of 1.5M (mean = 500k)
        {"New York", {1500000.0, 3}},
        // London has 100 samples, with a total price sum of 60M (mean = 600k)
        {"London", {60000000.0, 100}},
        // Tokyo has only one sample in our training set.
        {"Tokyo", {800000.0, 1}}
    };

    xt::transforms::target::LeaveOneOutEncoder encoder(city_stats, global_price_mean);
    std::cout.precision(10);
    std::cout << "Global Mean Price: " << global_price_mean << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing Leave-One-Out Encoding ---" << std::endl;

    // Case 1: A sample from "New York".
    // The other two samples had prices summing to 1.5M - 450k = 1.05M.
    // The LOO mean should be 1.05M / (3-1) = 525,000.
    test_encoder(encoder, "New York", 450000.0);

    // Case 2: A sample from the high-count "London" category.
    // The LOO mean should be very close to the category mean of 600k.
    test_encoder(encoder, "London", 750000.0);

    // Case 3: The single sample from "Tokyo".
    // Since count is 1, there are no "other" samples. It must fall back to the global mean.
    test_encoder(encoder, "Tokyo", 800000.0);

    // Case 4: An unseen city. This must also fall back to the global mean.
    test_encoder(encoder, "Paris", 650000.0);

    return 0;
}
*/

namespace xt::transforms::target {

    LeaveOneOutEncoder::LeaveOneOutEncoder(
        const std::unordered_map<std::string, TargetStats>& target_stats,
        double global_mean
    ) : target_stats_(target_stats), global_mean_(global_mean) {}

    auto LeaveOneOutEncoder::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("LeaveOneOutEncoder::forward requires two inputs: {label, target_value}.");
        }

        const std::any& label_any = any_vec[0];
        const std::any& target_any = any_vec[1];
        std::string key;
        double target_value;

        // Convert the input label to a string key.
        if (label_any.type() == typeid(std::string)) {
            key = std::any_cast<std::string>(label_any);
        } else if (label_any.type() == typeid(const char*)) {
            key = std::any_cast<const char*>(label_any);
        } else if (label_any.type() == typeid(int)) {
            key = std::to_string(std::any_cast<int>(label_any));
        } else if (label_any.type() == typeid(long)) {
            key = std::to_string(std::any_cast<long>(label_any));
        } else {
             throw std::invalid_argument("First input (label) must be a string or integer-like type.");
        }

        // Convert the input target value to a double.
        if (target_any.type() == typeid(double)) {
            target_value = std::any_cast<double>(target_any);
        } else if (target_any.type() == typeid(float)) {
            target_value = static_cast<double>(std::any_cast<float>(target_any));
        } else if (target_any.type() == typeid(int)) {
            target_value = static_cast<double>(std::any_cast<int>(target_any));
        } else if (target_any.type() == typeid(long)) {
            target_value = static_cast<double>(std::any_cast<long>(target_any));
        } else {
             throw std::invalid_argument("Second input (target_value) must be a numeric type.");
        }

        // 2. --- Core Logic ---
        auto it = target_stats_.find(key);

        if (it != target_stats_.end()) {
            // --- Category was seen during fitting ---
            const auto& stats = it->second;

            // CRITICAL EDGE CASE: If there was only one sample of this category in the
            // training set, there are no "other" samples to average.
            // In this case, the most robust estimate is the global mean.
            if (stats.count <= 1) {
                return global_mean_;
            }

            // The leave-one-out calculation.
            return (stats.sum - target_value) / (static_cast<double>(stats.count) - 1.0);
        } else {
            // --- Category is unseen ---
            // Fall back to the global mean.
            return global_mean_;
        }
    }

} // namespace xt::transforms::target