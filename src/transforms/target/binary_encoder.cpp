#include "include/transforms/target/binary_encoder.h"


#include <stdexcept>
#include <cmath> // For std::pow

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

int main() {
    // 1. --- Setup ---
    // We decide to encode our labels using 8 bits.
    // This means we can represent labels from 0 to 255.
    int num_bits = 8;
    xt::transforms::target::BinaryEncoder encoder(num_bits);
    std::cout << "BinaryEncoder created for " << num_bits << " bits." << std::endl;

    // 2. --- Define a sample label ---
    int label_index = 42;
    std::cout << "\nInput label: " << label_index << std::endl;
    // The binary representation of 42 is 101010.
    // In 8 bits, with MSB first, this is 00101010.

    // 3. --- Run the Transform ---
    auto binary_any = encoder.forward({label_index});

    // 4. --- Verify the Output ---
    try {
        auto binary_tensor = std::any_cast<torch::Tensor>(binary_any);

        std::cout << "Output binary tensor: " << binary_tensor << std::endl;

        // Expected output: [0., 0., 1., 0., 1., 0., 1., 0.]
        if (binary_tensor.size(0) == num_bits) {
            std::cout << "Verification successful!" << std::endl;
        }

    } catch (const std::bad_any_cast& e) {
        std::cerr << "Failed to cast result to torch::Tensor." << std::endl;
    }

    // --- Test Error Handling ---
    std::cout << "\n--- Testing Error Handling ---" << std::endl;
    int invalid_label = 300; // 300 is > 255, so it cannot be represented in 8 bits.
    std::cout << "Trying to encode invalid label: " << invalid_label << std::endl;
    try {
        encoder.forward({invalid_label});
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    BinaryEncoder::BinaryEncoder(int num_bits) : num_bits_(num_bits) {
        if (num_bits_ <= 0) {
            throw std::invalid_argument("Number of bits must be positive.");
        }
        if (num_bits_ > 63) {
            // Using a long for the label index limits us to ~63 bits.
            throw std::invalid_argument("Number of bits cannot exceed 63.");
        }
    }

    auto BinaryEncoder::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("BinaryEncoder::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        long label_index = 0;

        // Accept different integer types for flexibility.
        if (input_any.type() == typeid(int)) {
            label_index = std::any_cast<int>(input_any);
        } else if (input_any.type() == typeid(long)) {
            label_index = std::any_cast<long>(input_any);
        } else if (input_any.type() == typeid(short)) {
            label_index = std::any_cast<short>(input_any);
        } else {
             throw std::invalid_argument("Input to BinaryEncoder must be a scalar integer type (int, long, short).");
        }

        if (label_index < 0) {
            throw std::invalid_argument("Label index cannot be negative.");
        }

        // Check if the label can be represented with the given number of bits.
        long max_representable_value = (1L << num_bits_) - 1;
        if (label_index > max_representable_value) {
            throw std::invalid_argument(
                "Label index " + std::to_string(label_index) +
                " is too large to be represented with " + std::to_string(num_bits_) +
                " bits (max is " + std::to_string(max_representable_value) + ")."
            );
        }

        // 2. --- Create the Binary Tensor ---
        torch::Tensor binary_tensor = torch::zeros({num_bits_}, torch::kFloat32);

        // Fill the tensor from right to left (LSB to MSB)
        for (int i = num_bits_ - 1; i >= 0; --i) {
            if (label_index == 0) break; // Remainder of tensor will be zeros (correct)

            binary_tensor[i] = static_cast<float>(label_index % 2);
            label_index /= 2;
        }

        return binary_tensor;
    }

} // namespace xt::transforms::target