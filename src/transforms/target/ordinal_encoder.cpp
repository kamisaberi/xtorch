#include "include/transforms/target/ordinal_encoder.h"

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

void test_encoder(xt::transforms::target::OrdinalEncoder& encoder, const std::string& label) {
    std::cout << "Input: \"" << label << "\"";
    try {
        auto id_any = encoder.forward({label});
        auto id = std::any_cast<long>(id_any);
        std::cout << ", Output ID: " << id << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << ", Error: " << e.what() << std::endl;
    }
}

int main() {
    // 1. --- Setup ---
    // Define the explicit order of the categories.
    std::vector<std::string> ordered_sizes = {"small", "medium", "large", "x-large"};

    xt::transforms::target::OrdinalEncoder encoder(ordered_sizes);
    std::cout << "OrdinalEncoder created with order: small(0), medium(1), large(2), x-large(3)" << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing Ordinal Encoding ---" << std::endl;

    test_encoder(encoder, "medium"); // Expected: 1
    test_encoder(encoder, "small");  // Expected: 0
    test_encoder(encoder, "x-large"); // Expected: 3

    // --- Test Error Handling for an unseen label ---
    std::cout << "\n--- Testing Unseen Label ---" << std::endl;
    test_encoder(encoder, "tiny"); // Expected: Throws an error

    return 0;
}
*/

namespace xt::transforms::target {

    OrdinalEncoder::OrdinalEncoder(const std::vector<std::string>& ordered_categories) {
        if (ordered_categories.empty()) {
            throw std::invalid_argument("ordered_categories vector cannot be empty.");
        }

        // Build the mapping from the provided vector.
        // The index in the vector becomes the ordinal integer ID.
        for (long i = 0; i < ordered_categories.size(); ++i) {
            category_to_index_[ordered_categories[i]] = i;
        }
    }

    auto OrdinalEncoder::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("OrdinalEncoder::forward received an empty list.");
        }

        std::string key;
        try {
            key = std::any_cast<std::string>(any_vec[0]);
        } catch(const std::bad_any_cast&) {
            try {
                key = std::any_cast<const char*>(any_vec[0]);
            } catch(const std::bad_any_cast&) {
                throw std::invalid_argument("Input to OrdinalEncoder must be a std::string.");
            }
        }

        // 2. --- Core Logic: Lookup the ordinal index ---
        auto it = category_to_index_.find(key);

        if (it != category_to_index_.end()) {
            // Label was found, return its pre-defined ordinal index.
            return it->second;
        } else {
            // Unseen labels have no defined order in this scheme.
            // Throwing an error is the safest and most informative action.
            throw std::invalid_argument("Unseen label '" + key + "' encountered in OrdinalEncoder.");
        }
    }

} // namespace xt::transforms::target