#include "include/transforms/target/frequency_encoder.h"

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

void test_encoder(xt::transforms::target::FrequencyEncoder& encoder, const std::any& label) {
    if (label.type() == typeid(const char*)) {
         std::cout << "Input: \"" << std::any_cast<const char*>(label) << "\"";
    } else {
         std::cout << "Input: " << std::any_cast<int>(label);
    }

    auto freq_any = encoder.forward({label});
    auto freq = std::any_cast<double>(freq_any);
    std::cout << ", Output Frequency: " << freq << std::endl;
}

int main() {
    // 1. --- Setup ---
    // Imagine we analyzed a training dataset of 100 samples and found these counts:
    // - "cat": 50 times (0.5 frequency)
    // - "dog": 30 times (0.3 frequency)
    // - "bird": 20 times (0.2 frequency)
    std::unordered_map<std::string, float> animal_frequencies = {
        {"cat", 0.5f},
        {"dog", 0.3f},
        {"bird", 0.2f}
    };

    xt::transforms::target::FrequencyEncoder encoder(animal_frequencies);
    std::cout << "FrequencyEncoder created from pre-calculated data." << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing frequency encoding ---" << std::endl;

    // Case 1: A common category
    test_encoder(encoder, "cat"); // Expected: 0.5

    // Case 2: A less common category
    test_encoder(encoder, "dog"); // Expected: 0.3

    // Case 3: A label that was not in the training data (unseen)
    test_encoder(encoder, "fish"); // Expected: 0.0 (fallback)

    // --- Test with integer labels (converted to string for lookup) ---
    std::cout << "\n--- Testing with integer labels ---" << std::endl;
    std::unordered_map<std::string, float> id_frequencies = {
        {"101", 0.8f}, // Very common ID
        {"202", 0.2f}  // Rare ID
    };
    xt::transforms::target::FrequencyEncoder id_encoder(id_frequencies);
    test_encoder(id_encoder, 101); // Expected: 0.8
    test_encoder(id_encoder, 999); // Expected: 0.0 (unseen)

    return 0;
}
*/

namespace xt::transforms::target {

    FrequencyEncoder::FrequencyEncoder(const std::unordered_map<std::string, float>& frequency_map)
        : frequency_map_(frequency_map) {

        // Optional: Add validation for the provided frequencies
        for (const auto& pair : frequency_map_) {
            if (pair.second < 0.0f || pair.second > 1.0f) {
                throw std::invalid_argument(
                    "Frequency for label '" + pair.first + "' is " + std::to_string(pair.second) +
                    ", but must be between 0.0 and 1.0."
                );
            }
        }
    }

    auto FrequencyEncoder::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("FrequencyEncoder::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        std::string key;

        // 2. --- Handle different input types ---
        // Convert the input label, whatever its type, into a string key.
        if (input_any.type() == typeid(std::string)) {
            key = std::any_cast<std::string>(input_any);
        } else if (input_any.type() == typeid(const char*)) {
            key = std::any_cast<const char*>(input_any);
        } else if (input_any.type() == typeid(int)) {
            key = std::to_string(std::any_cast<int>(input_any));
        } else if (input_any.type() == typeid(long)) {
            key = std::to_string(std::any_cast<long>(input_any));
        } else {
             throw std::invalid_argument("Input to FrequencyEncoder must be a string or integer-like type.");
        }

        // 3. --- Core Logic: Lookup the frequency ---
        auto it = frequency_map_.find(key);

        if (it != frequency_map_.end()) {
            // Label was found, return its frequency.
            return static_cast<double>(it->second);
        } else {
            // Label was not seen during training, return 0.0 as a safe default.
            return 0.0;
        }
    }

} // namespace xt::transforms::target