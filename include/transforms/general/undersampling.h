#pragma once
#include "../common.h"

namespace xt::transforms::general {

    /**
     * @class UnderSampling
     * @brief A general transform that generates multiple sub-samples by repeatedly applying an extraction function.
     *
     * This transform takes a single tensor and applies a user-provided "cropper"
     * function multiple times to generate a stack of sub-samples. This is the
     * logical inverse of OverSampling.
     *
     * The provided function should implement the desired extraction logic,
     * such as random cropping or corner cropping.
     */
    struct UnderSampling final : xt::Module {
    public:
        /**
         * @brief Default constructor. Creates an uninitialized transform.
         */
        UnderSampling();

        /**
         * @brief Constructs the UnderSampling transform.
         * @param extractor A function (e.g., a lambda) that takes the source tensor
         *                  and returns one extracted sub-sample (e.g., a crop).
         * @param num_samples The number of sub-samples to generate.
         */
        UnderSampling(std::function<torch::Tensor(torch::Tensor)> extractor, int num_samples);

        /**
         * @brief Executes the extraction function multiple times and stacks the results.
         * @param tensors An initializer list expected to contain a single source tensor.
         * @return An std::any containing the resulting stacked torch::Tensor of
         *         shape [num_samples, ...].
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // A function that extracts one sub-sample from a source tensor.
        std::function<torch::Tensor(torch::Tensor)> extractor;
        int num_samples;
    };

} // namespace xt::transforms::general