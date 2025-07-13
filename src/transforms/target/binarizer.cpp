#include "include/transforms/target/binarizer.h"

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

int main() {
    // 1. --- Setup ---
    // We have a classification problem with 5 classes (indexed 0 through 4).
    int num_classes = 5;
    xt::transforms::target::Binarizer binarizer(num_classes);
    std::cout << "Binarizer created for " << num_classes << " classes." << std::endl;

    // 2. --- Define a sample label ---
    int label_index = 3;
    std::cout << "\nInput label: " << label_index << std::endl;

    // 3. --- Run the Transform ---
    auto binarized_any = binarizer.forward({label_index});

    // 4. --- Verify the Output ---
    try {
        auto binarized_tensor = std::any_cast<torch::Tensor>(binarized_any);

        std::cout << "Output one-hot tensor: " << binarized_tensor << std::endl;

        // Expected output: [0., 0., 0., 1., 0.]
        if (binarized_tensor.size(0) == 5 && binarized_tensor[3].item<float>() == 1.0f) {
            std::cout << "Verification successful!" << std::endl;
        }

    } catch (const std::bad_any_cast& e) {
        std::cerr << "Failed to cast result to torch::Tensor." << std::endl;
    }

    // --- Test Error Handling ---
    std::cout << "\n--- Testing Error Handling ---" << std::endl;
    int invalid_label = 5;
    std::cout << "Trying to binarize invalid label: " << invalid_label << std::endl;
    try {
        binarizer.forward({invalid_label});
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    Binarizer::Binarizer(int num_classes) : num_classes_(num_classes) {
        if (num_classes_ <= 0) {
            throw std::invalid_argument("Number of classes must be positive.");
        }
    }

    auto Binarizer::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Binarizer::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        long label_index = 0;

        // Make the transform robust by accepting different integer types.
        if (input_any.type() == typeid(int)) {
            label_index = std::any_cast<int>(input_any);
        } else if (input_any.type() == typeid(long)) {
            label_index = std::any_cast<long>(input_any);
        } else if (input_any.type() == typeid(short)) {
            label_index = std::any_cast<short>(input_any);
        } else {
             throw std::invalid_argument("Input to Binarizer must be a scalar integer type (int, long, short).");
        }

        // Validate that the label is in the correct range [0, num_classes-1].
        if (label_index < 0 || label_index >= num_classes_) {
            throw std::invalid_argument(
                "Label index " + std::to_string(label_index) + " is out of bounds for " +
                std::to_string(num_classes_) + " classes."
            );
        }

        // 2. --- Create the One-Hot Tensor ---
        // Create a 1D tensor of zeros with length equal to the number of classes.
        torch::Tensor one_hot_tensor = torch::zeros({num_classes_}, torch::kFloat32);

        // Set the value at the specified index to 1.0.
        one_hot_tensor[label_index] = 1.0f;

        return one_hot_tensor;
    }

} // namespace xt::transforms::target