#include <transforms/target/label_smoother.h>

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
    int num_classes = 5;
    float epsilon = 0.1f;

    // The target for the correct class should be 1.0 - 0.1 = 0.9
    // The target for each incorrect class should be 0.1 / (5 - 1) = 0.1 / 4 = 0.025
    xt::transforms::target::LabelSmoother smoother(num_classes, epsilon);
    std::cout << "LabelSmoother created for " << num_classes << " classes with epsilon=" << epsilon << "." << std::endl;

    // 2. --- Define a sample label ---
    int label_index = 2;
    std::cout << "\nInput label: " << label_index << std::endl;

    // 3. --- Run the Transform ---
    auto smoothed_any = smoother.forward({label_index});

    // 4. --- Verify the Output ---
    try {
        auto smoothed_tensor = std::any_cast<torch::Tensor>(smoothed_any);

        std::cout << "Output smoothed tensor: " << smoothed_tensor << std::endl;
        std::cout << "Sum of tensor values: " << smoothed_tensor.sum().item<float>() << " (should be 1.0)" << std::endl;

        // Expected output: [0.025, 0.025, 0.9, 0.025, 0.025]
        if (torch::allclose(smoothed_tensor.sum(), torch::tensor(1.0f))) {
            std::cout << "Verification successful!" << std::endl;
        }

    } catch (const std::bad_any_cast& e) {
        std::cerr << "Failed to cast result to torch::Tensor." << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    LabelSmoother::LabelSmoother(int num_classes, float epsilon)
        : num_classes_(num_classes), epsilon_(epsilon) {

        if (num_classes_ <= 1) {
            throw std::invalid_argument("num_classes must be greater than 1 for label smoothing.");
        }
        if (epsilon_ < 0.0f || epsilon_ >= 1.0f) {
            throw std::invalid_argument("Epsilon must be in the range [0.0, 1.0).");
        }

        // Pre-calculate the two values we will use to fill the tensor.
        value_for_correct_class_ = 1.0f - epsilon_;
        value_for_incorrect_class_ = epsilon_ / (num_classes_ - 1);
    }

    auto LabelSmoother::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("LabelSmoother::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        long correct_label_idx = 0;

        if (input_any.type() == typeid(int)) {
            correct_label_idx = std::any_cast<int>(input_any);
        } else if (input_any.type() == typeid(long)) {
            correct_label_idx = std::any_cast<long>(input_any);
        } else {
             throw std::invalid_argument("Input to LabelSmoother must be a scalar integer type.");
        }

        if (correct_label_idx < 0 || correct_label_idx >= num_classes_) {
            throw std::invalid_argument("Label index is out of bounds for the number of classes.");
        }

        // 2. --- Core Logic ---
        // Create a tensor filled entirely with the "incorrect" value. This is efficient.
        torch::Tensor smoothed_labels = torch::full(
            {num_classes_},
            value_for_incorrect_class_,
            torch::kFloat32
        );

        // Overwrite the single correct index with the "correct" smoothed value.
        smoothed_labels[correct_label_idx] = value_for_correct_class_;

        return smoothed_labels;
    }

} // namespace xt::transforms::target