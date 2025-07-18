#include <transforms/general/oversampling.h>
#include <vector>
#include <stdexcept>

namespace xt::transforms::general {

    // Default constructor
    OverSampling::OverSampling() = default;

    /**
     * @brief Constructs an OverSampling transform.
     * @param transform A function (like a lambda) that takes a tensor and returns a transformed tensor.
     * @param num_samples The number of times to apply the transform to create augmented samples.
     */
    OverSampling::OverSampling(std::function<torch::Tensor(torch::Tensor)> transform, int num_samples)
        : xt::Module(), transform(transform), num_samples(num_samples) {

        // Ensure num_samples is a positive integer
        if (num_samples <= 0) {
            throw std::invalid_argument("OverSampling requires num_samples to be a positive integer.");
        }
    }

    /**
     * @brief Applies a transformation multiple times to an input tensor and stacks the results.
     * @param tensors An initializer list containing the tensor to be oversampled.
     * @return A single tensor where the first dimension is the number of samples,
     *         i.e., (num_samples, C, H, W), wrapped in std::any.
     */
    auto OverSampling::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. Convert the initializer_list to a std::vector to safely access elements.
        std::vector<std::any> any_vec(tensors);

        if (any_vec.empty()) {
            throw std::invalid_argument("OverSampling::forward received an empty list of tensors.");
        }

        // 2. Safely cast the first element from std::any to torch::Tensor.
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to OverSampling is not defined.");
        }

        // 3. Ensure a transform function has been provided.
        if (!transform) {
            throw std::runtime_error("OverSampling transform function is not set.");
        }

        // 4. Create a vector to hold the augmented tensors.
        std::vector<torch::Tensor> augmented_tensors;
        augmented_tensors.reserve(num_samples); // Pre-allocate memory for efficiency

        // 5. Loop to generate the samples.
        for (int i = 0; i < num_samples; ++i) {
            // Apply the provided transformation to a clone of the input tensor.
            // Using a clone is important if the transform modifies the tensor in-place.
            augmented_tensors.push_back(transform(input_tensor.clone()));
        }

        // 6. Stack the list of tensors into a single new tensor along a new dimension (dim=0).
        // The result will have the shape [num_samples, original_shape...].
        return torch::stack(augmented_tensors, 0);
    }

} // namespace xt::transforms::general