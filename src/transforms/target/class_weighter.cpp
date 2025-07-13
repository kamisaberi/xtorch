#include "include/transforms/target/class_weighter.h"

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
    // Define weights for a 4-class imbalanced problem.
    // Let's say class 2 is rare, so we give it a higher weight.
    std::vector<float> weights = {0.5f, 0.8f, 3.0f, 0.8f};

    xt::transforms::target::ClassWeighter weighter(weights);
    std::cout << "ClassWeighter created for " << weights.size() << " classes." << std::endl;
    std::cout << "Weights: [0.5, 0.8, 3.0, 0.8]" << std::endl;

    // 2. --- Define a sample label from the rare class ---
    int label = 2;
    std::cout << "\nInput label: " << label << std::endl;

    // 3. --- Run the Transform ---
    auto weight_any = weighter.forward({label});

    // 4. --- Verify the Output ---
    try {
        auto weight_tensor = std::any_cast<torch::Tensor>(weight_any);

        // .item<float>() is the correct way to get the value from a scalar tensor
        std::cout << "Output weight (as scalar tensor): " << weight_tensor << std::endl;
        std::cout << "Output weight (as float): " << weight_tensor.item<float>() << std::endl;

        // Expected output: 3.0
        if (weight_tensor.item<float>() == 3.0f) {
            std::cout << "Verification successful!" << std::endl;
        }

    } catch (const std::bad_any_cast& e) {
        std::cerr << "Failed to cast result to torch::Tensor." << std::endl;
    }

    // --- Test Error Handling ---
    std::cout << "\n--- Testing Error Handling ---" << std::endl;
    int invalid_label = 4; // Our weights vector only has indices 0-3.
    std::cout << "Trying to get weight for invalid label: " << invalid_label << std::endl;
    try {
        weighter.forward({invalid_label});
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    ClassWeighter::ClassWeighter(const std::vector<float>& class_weights)
        : class_weights_(class_weights), num_classes_(class_weights.size()) {

        if (class_weights_.empty()) {
            throw std::invalid_argument("Class weights vector cannot be empty.");
        }
    }

    auto ClassWeighter::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("ClassWeighter::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        long class_index = 0;

        // Handle various integer types for the input label
        if (input_any.type() == typeid(int)) {
            class_index = std::any_cast<int>(input_any);
        } else if (input_any.type() == typeid(long)) {
            class_index = std::any_cast<long>(input_any);
        } else if (input_any.type() == typeid(short)) {
            class_index = std::any_cast<short>(input_any);
        } else {
             throw std::invalid_argument("Input to ClassWeighter must be a scalar integer type (int, long, short).");
        }

        // Check if the provided class index is valid for our weights vector.
        if (class_index < 0 || class_index >= num_classes_) {
            throw std::invalid_argument(
                "Class index " + std::to_string(class_index) + " is out of bounds for the " +
                std::to_string(num_classes_) + " provided weights."
            );
        }

        // 2. --- Look up the weight ---
        float weight = class_weights_[class_index];

        // 3. --- Create and return a scalar tensor ---
        // A scalar tensor is 0-dimensional and is ideal for multiplying with a loss value.
        return torch::tensor(weight, torch::kFloat32);
    }

} // namespace xt::transforms::target