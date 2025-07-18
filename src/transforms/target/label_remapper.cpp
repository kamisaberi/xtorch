#include <transforms/target/label_remapper.h>

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

void test_remapper(xt::transforms::target::LabelRemapper& remapper, long input_label) {
    std::cout << "Input: " << input_label;
    auto remapped_any = remapper.forward({input_label});
    auto remapped_label = std::any_cast<long>(remapped_any);
    std::cout << ", Output: " << remapped_label << std::endl;
}

int main() {
    // 1. --- Setup ---
    // Define a map to merge classes. Let's say we want to:
    // - Map class 3 (e.g., "sedan") to class 1 ("car").
    // - Map class 4 (e.g., "suv") to class 1 ("car").
    // - Map class 8 (e.g., "bicycle") to class 7 ("cycle").
    std::unordered_map<long, long> remapping_rules = {
        {3, 1},
        {4, 1},
        {8, 7}
    };

    xt::transforms::target::LabelRemapper remapper(remapping_rules);
    std::cout << "LabelRemapper created with specific rules." << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing remapping ---" << std::endl;

    // Case 1: A label that should be remapped
    test_remapper(remapper, 3); // Expected: 1

    // Case 2: Another label that should be remapped to the same new class
    test_remapper(remapper, 4); // Expected: 1

    // Case 3: A label that should be remapped to a different class
    test_remapper(remapper, 8); // Expected: 7

    // Case 4: A label that is NOT in the map and should be passed through
    test_remapper(remapper, 2); // Expected: 2

    return 0;
}
*/

namespace xt::transforms::target {

    LabelRemapper::LabelRemapper(const std::unordered_map<long, long>& remapping_map)
        : remapping_map_(remapping_map) {}

    auto LabelRemapper::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("LabelRemapper::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        long original_label = 0;

        // Handle various integer types
        if (input_any.type() == typeid(int)) {
            original_label = std::any_cast<int>(input_any);
        } else if (input_any.type() == typeid(long)) {
            original_label = std::any_cast<long>(input_any);
        } else if (input_any.type() == typeid(short)) {
            original_label = std::any_cast<short>(input_any);
        } else {
             throw std::invalid_argument("Input to LabelRemapper must be a scalar integer type.");
        }

        // 2. --- Core Logic: Lookup and Remap ---
        auto it = remapping_map_.find(original_label);

        if (it != remapping_map_.end()) {
            // The label was found in the map, so return the new, remapped value.
            return it->second;
        } else {
            // The label was not found in the map, so pass it through unchanged.
            return original_label;
        }
    }

} // namespace xt::transforms::target