#include <transforms/target/hashing_encoder.h>

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

// Helper to find the index of the '1' in the output tensor
long find_active_index(const torch::Tensor& tensor) {
    return torch::argmax(tensor).item<long>();
}

void test_encoder(xt::transforms::target::HashingEncoder& encoder, const std::any& label, int n_dims) {
    std::string key;
     if (label.type() == typeid(const char*)) {
         key = std::any_cast<const char*>(label);
         std::cout << "Input: \"" << key << "\"";
    } else {
         key = std::to_string(std::any_cast<int>(label));
         std::cout << "Input: " << key;
    }

    auto hashed_any = encoder.forward({label});
    auto hashed_tensor = std::any_cast<torch::Tensor>(hashed_any);
    long index = find_active_index(hashed_tensor);

    // For verification, let's calculate the expected index manually
    long expected_index = std::hash<std::string>{}(key) % n_dims;

    std::cout << ", Hashed Index: " << index << " (Expected: " << expected_index << ")" << std::endl;
}

int main() {
    // 1. --- Setup ---
    // We want to map our categories into an 8-dimensional space.
    int n_dims = 8;
    xt::transforms::target::HashingEncoder encoder(n_dims);
    std::cout << "HashingEncoder created for output dimension " << n_dims << "." << std::endl;

    // 2. --- Test Cases ---
    std::cout << "\n--- Testing hashing ---" << std::endl;

    test_encoder(encoder, "cat", n_dims);
    test_encoder(encoder, "dog", n_dims);
    test_encoder(encoder, "a_very_long_category_name", n_dims);

    // --- Test with an integer label ---
    test_encoder(encoder, 12345, n_dims);

    // --- Test for hash collision ---
    // With a small dimension like 8, it's easy to find a collision.
    std::cout << "\n--- Checking for a hash collision ---" << std::endl;
    xt::transforms::target::HashingEncoder collision_checker(4);
    test_encoder(collision_checker, "apple", 4);
    test_encoder(collision_checker, "orange", 4);
    // Note: The hash values for "apple" and "orange" might or might not collide depending
    // on the C++ standard library implementation, but this shows how to test it.

    return 0;
}
*/

namespace xt::transforms::target {

    HashingEncoder::HashingEncoder(int num_dimensions)
        : num_dimensions_(num_dimensions) {

        if (num_dimensions_ <= 0) {
            throw std::invalid_argument("Number of dimensions must be positive.");
        }
    }

    auto HashingEncoder::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("HashingEncoder::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        std::string key;

        // 2. --- Convert input label to a string key ---
        if (input_any.type() == typeid(std::string)) {
            key = std::any_cast<std::string>(input_any);
        } else if (input_any.type() == typeid(const char*)) {
            key = std::any_cast<const char*>(input_any);
        } else if (input_any.type() == typeid(int)) {
            key = std::to_string(std::any_cast<int>(input_any));
        } else if (input_any.type() == typeid(long)) {
            key = std::to_string(std::any_cast<long>(input_any));
        } else {
             throw std::invalid_argument("Input to HashingEncoder must be a string or integer-like type.");
        }

        // 3. --- Core Logic: Hash and Map to Index ---
        // Use the standard library's hash function to get a hash value.
        size_t hash_value = hasher_(key);

        // Modulo the hash value by the number of dimensions to get a valid index.
        // The result is guaranteed to be in the range [0, num_dimensions - 1].
        long index = hash_value % num_dimensions_;

        // 4. --- Create the Output Tensor ---
        // Create a zero vector and place a '1' at the calculated index.
        torch::Tensor hashed_tensor = torch::zeros({num_dimensions_}, torch::kFloat32);
        hashed_tensor[index] = 1.0f;

        return hashed_tensor;
    }

} // namespace xt::transforms::target