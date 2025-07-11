#include "include/transforms/general/undersampling.h"


namespace xt::transforms::general {

    // Default constructor
    UnderSampling::UnderSampling() = default;

    /**
     * @brief Constructs an UnderSampling transform.
     * @param extractor A function that takes a source tensor and returns one sub-sample.
     * @param num_samples The number of times to call the extractor function.
     */
    UnderSampling::UnderSampling(std::function<torch::Tensor(torch::Tensor)> extractor, int num_samples)
        : xt::Module(), extractor(extractor), num_samples(num_samples) {

        if (num_samples <= 0) {
            throw std::invalid_argument("UnderSampling requires num_samples to be a positive integer.");
        }
    }

    /**
     * @brief Applies an extraction function multiple times to an input tensor and stacks the results.
     * @param tensors An initializer list containing the tensor to be under-sampled.
     * @return A single tensor where the first dimension is the number of samples,
     *         i.e., (num_samples, C, crop_H, crop_W), wrapped in std::any.
     */
    auto UnderSampling::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. Convert the initializer_list to a std::vector to safely access elements.
        std::vector<std::any> any_vec(tensors);

        if (any_vec.empty()) {
            throw std::invalid_argument("UnderSampling::forward received an empty list of tensors.");
        }

        // 2. Safely cast the first element from std::any to torch::Tensor.
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to UnderSampling is not defined.");
        }

        // 3. Ensure an extractor function has been provided.
        if (!extractor) {
            throw std::runtime_error("UnderSampling extractor function is not set.");
        }

        // 4. Create a vector to hold the extracted sub-sample tensors.
        std::vector<torch::Tensor> extracted_tensors;
        extracted_tensors.reserve(num_samples); // Pre-allocate memory for efficiency

        // 5. Loop to generate the samples.
        for (int i = 0; i < num_samples; ++i) {
            // Apply the provided extraction function to the input tensor.
            // Unlike OverSampling's transform, the extractor shouldn't modify the input,
            // so cloning isn't strictly necessary but is still a safe practice.
            extracted_tensors.push_back(extractor(input_tensor));
        }

        // 6. Stack the list of tensors into a single new tensor along a new dimension (dim=0).
        // The result will have the shape [num_samples, ...].
        return torch::stack(extracted_tensors, 0);
    }

} // namespace xt::transforms::general