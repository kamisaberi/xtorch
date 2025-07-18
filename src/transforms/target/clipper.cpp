#include <transforms/target/clipper.h>

#include <stdexcept>
#include <algorithm> // For std::min and std::max

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

void test_clipping(xt::transforms::target::Clipper& clipper, long input_label) {
    std::cout << "Input: " << input_label;
    auto clipped_any = clipper.forward({input_label});
    auto clipped_label = std::any_cast<long>(clipped_any);
    std::cout << ", Output: " << clipped_label << std::endl;
}

int main() {
    // 1. --- Setup ---
    // Create a clipper that forces labels to be between 10 and 20.
    long min_val = 10;
    long max_val = 20;
    xt::transforms::target::Clipper clipper(min_val, max_val);
    std::cout << "Clipper created for range [" << min_val << ", " << max_val << "]" << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing clipping ---" << std::endl;

    // Case 1: Value below the minimum
    test_clipping(clipper, 5); // Expected: 10

    // Case 2: Value within the range
    test_clipping(clipper, 15); // Expected: 15

    // Case 3: Value above the maximum
    test_clipping(clipper, 25); // Expected: 20

    // --- Test Error Handling ---
    std::cout << "\n--- Testing invalid constructor ---" << std::endl;
    try {
        xt::transforms::target::Clipper bad_clipper(20, 10); // min > max
    } catch(const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    Clipper::Clipper(long min_val, long max_val)
        : min_val_(min_val), max_val_(max_val) {

        if (min_val_ > max_val_) {
            throw std::invalid_argument("In Clipper, min_val cannot be greater than max_val.");
        }
    }

    auto Clipper::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Clipper::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        long input_label = 0;

        // Handle various integer types for the input label
        if (input_any.type() == typeid(int)) {
            input_label = std::any_cast<int>(input_any);
        } else if (input_any.type() == typeid(long)) {
            input_label = std::any_cast<long>(input_any);
        } else if (input_any.type() == typeid(short)) {
            input_label = std::any_cast<short>(input_any);
        } else {
             throw std::invalid_argument("Input to Clipper must be a scalar integer type (int, long, short).");
        }

        // 2. --- Core Logic: Clamp the value ---
        // This is the most efficient way to clamp a value to a range.
        // First, ensure it's not greater than the max.
        // Then, ensure the result is not less than the min.
        long clipped_label = std::max(min_val_, std::min(input_label, max_val_));

        // 3. --- Return the result ---
        return clipped_label;
    }

} // namespace xt::transforms::target