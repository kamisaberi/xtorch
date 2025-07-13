#include "include/transforms/target/label_encoder.h"

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

void test_encoder(xt::transforms::target::LabelEncoder& encoder, const std::string& label) {
    auto id_any = encoder.forward({label});
    auto id = std::any_cast<long>(id_any);
    std::cout << "Input: \"" << label << "\", Output ID: " << id << std::endl;
}

int main() {
    // 1. --- Setup ---
    xt::transforms::target::LabelEncoder encoder;
    std::cout << "LabelEncoder created. It is currently 'un-fitted'." << std::endl;

    // 2. --- Run a sequence of labels ---
    // This demonstrates the stateful nature of the encoder.
    std::cout << "\n--- Encoding a sequence of labels ---" << std::endl;

    test_encoder(encoder, "cat");      // Should assign ID 0
    test_encoder(encoder, "dog");      // Should assign ID 1
    test_encoder(encoder, "cat");      // Should return existing ID 0
    test_encoder(encoder, "bird");     // Should assign ID 2
    test_encoder(encoder, "dog");      // Should return existing ID 1
    test_encoder(encoder, "cat");      // Should return existing ID 0

    // 3. --- Inspect the learned vocabulary ---
    std::cout << "\n--- Inspecting the final learned mapping ---" << std::endl;
    auto learned_map = encoder.get_mapping();
    for (const auto& pair : learned_map) {
        std::cout << "Label: \"" << pair.first << "\" -> ID: " << pair.second << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    LabelEncoder::LabelEncoder() : next_id_(0) {}

    auto LabelEncoder::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("LabelEncoder::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        std::string key;

        // Convert the input label to a string key for the map.
        if (input_any.type() == typeid(std::string)) {
            key = std::any_cast<std::string>(input_any);
        } else if (input_any.type() == typeid(const char*)) {
            key = std::any_cast<const char*>(input_any);
        } else if (input_any.type() == typeid(int)) {
            key = std::to_string(std::any_cast<int>(input_any));
        } else if (input_any.type() == typeid(long)) {
            key = std::to_string(std::any_cast<long>(input_any));
        } else {
             throw std::invalid_argument("Input to LabelEncoder must be a string or integer-like type.");
        }

        // 2. --- Core Stateful Logic ---
        auto it = mapping_.find(key);

        if (it != mapping_.end()) {
            // --- Label has been seen before ---
            // Return the existing ID from the map.
            return it->second;
        } else {
            // --- Label is new ---
            // Assign the next available ID, store it in the map, and increment the ID counter.
            long new_id = next_id_;
            mapping_[key] = new_id;
            next_id_++;
            return new_id;
        }
    }

    auto LabelEncoder::get_mapping() const -> std::unordered_map<std::string, long> {
        return mapping_;
    }

} // namespace xt::transforms::target