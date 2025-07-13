#include "include/transforms/text/pad_transform.h"


// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

// Helper to print a vector
void print_vector(const std::string& name, const std::vector<long>& vec) {
    std::cout << name << " (size " << vec.size() << "): [ ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    // --- Test Data ---
    std::vector<long> short_sequence = {101, 2054, 2003, 102}; // len=4
    std::vector<long> long_sequence = {101, 7592, 1010, 2026, 3899, 2001, 102}; // len=7
    int max_len = 6;

    std::cout << "--- Max Length set to " << max_len << " ---" << std::endl;
    print_vector("\nShort Sequence", short_sequence);
    print_vector("Long Sequence ", long_sequence);

    // --- Example 1: Standard Right Padding (on short sequence) ---
    std::cout << "\n--- 1. Testing Right Padding ---" << std::endl;
    xt::transforms::text::PadTransform right_padder(max_len, 0); // pad_id = 0
    auto padded_right = std::any_cast<std::vector<long>>(right_padder.forward({short_sequence}));
    print_vector("Result", padded_right);

    // --- Example 2: Left Padding (on short sequence) ---
    std::cout << "\n--- 2. Testing Left Padding ---" << std::endl;
    xt::transforms::text::PadTransform left_padder(max_len, 0, xt::transforms::text::PaddingDirection::LEFT);
    auto padded_left = std::any_cast<std::vector<long>>(left_padder.forward({short_sequence}));
    print_vector("Result", padded_left);

    // --- Example 3: Standard Right Truncation (on long sequence) ---
    std::cout << "\n--- 3. Testing Right Truncation ---" << std::endl;
    auto truncated_right = std::any_cast<std::vector<long>>(right_padder.forward({long_sequence}));
    print_vector("Result", truncated_right);

    // --- Example 4: Left Truncation (on long sequence) ---
    std::cout << "\n--- 4. Testing Left Truncation ---" << std::endl;
    xt::transforms::text::PadTransform left_truncator(max_len, 0, xt::transforms::text::PaddingDirection::RIGHT, xt::transforms::text::TruncationDirection::LEFT);
    auto truncated_left = std::any_cast<std::vector<long>>(left_truncator.forward({long_sequence}));
    print_vector("Result", truncated_left);

    return 0;
}
*/

namespace xt::transforms::text {

    PadTransform::PadTransform(int max_len, long pad_id, PaddingDirectionP pad_dir, TruncationDirectionP trunc_dir)
            : max_len_(max_len), pad_id_(pad_id), pad_dir_(pad_dir), trunc_dir_(trunc_dir) {
        if (max_len_ <= 0) {
            throw std::invalid_argument("PadTransform max_len must be positive.");
        }
    }

    auto PadTransform::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("PadTransform::forward received an empty list.");
        }

        std::vector<long> input_vec;
        try {
            input_vec = std::any_cast<std::vector<long>>(any_vec[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Input to PadTransform must be of type std::vector<long>.");
        }

        // Make a copy to modify
        std::vector<long> output_vec = input_vec;

        // 2. --- Truncation (if sequence is too long) ---
        if (output_vec.size() > max_len_) {
            if (trunc_dir_ == TruncationDirectionP::RIGHT) {
                output_vec.resize(max_len_);
            } else { // TruncationDirection::LEFT
                output_vec.erase(output_vec.begin(), output_vec.begin() + (output_vec.size() - max_len_));
            }
        }

            // 3. --- Padding (if sequence is too short) ---
        else if (output_vec.size() < max_len_) {
            size_t num_pads = max_len_ - output_vec.size();
            if (pad_dir_ == PaddingDirectionP::RIGHT) {
                output_vec.insert(output_vec.end(), num_pads, pad_id_);
            } else { // PaddingDirection::LEFT
                output_vec.insert(output_vec.begin(), num_pads, pad_id_);
            }
        }

        // 4. --- Return the fixed-size vector ---
        return output_vec;
    }

} // namespace xt::transforms::text