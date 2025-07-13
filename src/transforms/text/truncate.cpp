#include "include/transforms/text/truncate.h"

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

// Helper to print a vector
template<typename T>
void print_vector(const std::string& name, const std::vector<T>& vec) {
    std::cout << name << " (size " << vec.size() << "): [ ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    // --- Test Data ---
    std::vector<std::string> long_tokens = {"[CLS]", "this", "is", "a", "very", "long", "sentence", "[SEP]"}; // len=8
    std::vector<long> long_ids = {101, 1045, 2003, 1037, 2200, 2146, 6251, 102}; // len=8
    int max_len = 5;

    std::cout << "--- Max Length set to " << max_len << " ---" << std::endl;
    print_vector("\nOriginal Tokens", long_tokens);
    print_vector("Original IDs   ", long_ids);

    // --- Example 1: Standard Right Truncation (default) ---
    std::cout << "\n--- 1. Testing Right Truncation ---" << std::endl;
    xt::transforms::text::Truncate right_truncator(max_len);

    auto truncated_tokens_right = std::any_cast<std::vector<std::string>>(right_truncator.forward({long_tokens}));
    print_vector("Result (Tokens)", truncated_tokens_right);

    auto truncated_ids_right = std::any_cast<std::vector<long>>(right_truncator.forward({long_ids}));
    print_vector("Result (IDs)   ", truncated_ids_right);

    // --- Example 2: Left Truncation ---
    std::cout << "\n--- 2. Testing Left Truncation ---" << std::endl;
    xt::transforms::text::Truncate left_truncator(max_len, xt::transforms::text::TruncationDirection::LEFT);

    auto truncated_tokens_left = std::any_cast<std::vector<std::string>>(left_truncator.forward({long_tokens}));
    print_vector("Result (Tokens)", truncated_tokens_left);

    auto truncated_ids_left = std::any_cast<std::vector<long>>(left_truncator.forward({long_ids}));
    print_vector("Result (IDs)   ", truncated_ids_left);

    return 0;
}
*/

namespace xt::transforms::text {

    Truncate::Truncate(int max_len, TruncationDirection trunc_dir)
            : max_len_(max_len), trunc_dir_(trunc_dir) {
        if (max_len_ < 0) {
            throw std::invalid_argument("Truncate max_len cannot be negative.");
        }
    }

    auto Truncate::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Truncate::forward received an empty list.");
        }
        const std::any& input_any = any_vec[0];

        // 2. --- Handle Multiple Input Types ---
        // This transform can operate on token IDs (vector<long>) or string tokens (vector<string>).

        // Try casting to vector<long> first
        try {
            std::vector<long> vec = std::any_cast<std::vector<long>>(input_any);
            if (vec.size() > max_len_) {
                if (trunc_dir_ == TruncationDirection::RIGHT) {
                    vec.resize(max_len_);
                } else { // TruncationDirection::LEFT
                    vec.erase(vec.begin(), vec.begin() + (vec.size() - max_len_));
                }
            }
            return vec;
        } catch (const std::bad_any_cast&) {
            // If the first cast fails, ignore the exception and try the next type.
        }

        // Try casting to vector<string>
        try {
            std::vector<std::string> vec = std::any_cast<std::vector<std::string>>(input_any);
            if (vec.size() > max_len_) {
                if (trunc_dir_ == TruncationDirection::RIGHT) {
                    vec.resize(max_len_);
                } else { // TruncationDirection::LEFT
                    vec.erase(vec.begin(), vec.begin() + (vec.size() - max_len_));
                }
            }
            return vec;
        } catch (const std::bad_any_cast&) {
            // If this also fails, then the input type is unsupported.
        }

        // 3. --- Throw error if type is unsupported ---
        throw std::invalid_argument("Input to Truncate must be of type std::vector<long> or std::vector<std::string>.");
    }

} // namespace xt::transforms::text