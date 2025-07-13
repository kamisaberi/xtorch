#include "include/transforms/target/discretizer.h"

#include <stdexcept>
#include <algorithm> // For std::is_sorted

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

void test_discretizer(xt::transforms::target::Discretizer& discretizer, float age) {
    std::cout << "Input age: " << age;
    auto bin_any = discretizer.forward({age});
    auto bin_index = std::any_cast<long>(bin_any);
    std::cout << ", Output bin: " << bin_index << std::endl;
}

int main() {
    // 1. --- Setup ---
    // We want to bin ages into 4 categories:
    // Bin 0: "Child" (< 18)
    // Bin 1: "Young Adult" (18-34)
    // Bin 2: "Adult" (35-59)
    // Bin 3: "Senior" (60+)
    // The boundaries are therefore 18, 35, and 60.
    std::vector<float> age_boundaries = {18.0f, 35.0f, 60.0f};

    xt::transforms::target::Discretizer age_discretizer(age_boundaries);
    std::cout << "Discretizer created with boundaries [18, 35, 60]." << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing discretization ---" << std::endl;

    // Test a value in each bin
    test_discretizer(age_discretizer, 10.5f); // Expected bin: 0
    test_discretizer(age_discretizer, 25.0f); // Expected bin: 1
    test_discretizer(age_discretizer, 45.0f); // Expected bin: 2
    test_discretizer(age_discretizer, 70.0f); // Expected bin: 3

    // Test a value exactly on a boundary (should go into the higher bin)
    std::cout << "\n--- Testing boundary condition ---" << std::endl;
    test_discretizer(age_discretizer, 18.0f); // Expected bin: 1

    // --- Test Error Handling ---
    std::cout << "\n--- Testing invalid constructor ---" << std::endl;
    try {
        // Unsorted boundaries should throw an error.
        xt::transforms::target::Discretizer bad_discretizer({30.0f, 10.0f, 20.0f});
    } catch(const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    Discretizer::Discretizer(const std::vector<float>& bin_boundaries)
        : bin_boundaries_(bin_boundaries) {

        if (bin_boundaries_.empty()) {
            throw std::invalid_argument("Bin boundaries vector cannot be empty.");
        }
        if (!std::is_sorted(bin_boundaries_.begin(), bin_boundaries_.end())) {
            throw std::invalid_argument("Bin boundaries must be sorted in ascending order.");
        }
    }

    auto Discretizer::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Discretizer::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        float input_value = 0.0f;

        // Handle various numeric types by casting to float.
        if (input_any.type() == typeid(float)) {
            input_value = std::any_cast<float>(input_any);
        } else if (input_any.type() == typeid(double)) {
            input_value = static_cast<float>(std::any_cast<double>(input_any));
        } else if (input_any.type() == typeid(int)) {
            input_value = static_cast<float>(std::any_cast<int>(input_any));
        } else if (input_any.type() == typeid(long)) {
            input_value = static_cast<float>(std::any_cast<long>(input_any));
        } else {
             throw std::invalid_argument("Input to Discretizer must be a scalar numeric type.");
        }

        // 2. --- Core Logic: Find the correct bin ---
        // Find the first boundary that is greater than the input value.
        // The position of this boundary is the bin index.
        for (size_t i = 0; i < bin_boundaries_.size(); ++i) {
            if (input_value < bin_boundaries_[i]) {
                return static_cast<long>(i);
            }
        }

        // If the loop completes, the value is greater than or equal to all
        // boundaries, so it belongs in the last bin. The index of this bin
        // is equal to the number of boundaries.
        return static_cast<long>(bin_boundaries_.size());
    }

} // namespace xt::transforms::target