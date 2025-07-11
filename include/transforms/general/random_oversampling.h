#pragma once

#include "../common.h"


namespace xt::transforms::general {

    /**
     * @class RandomOverSampling
     * @brief A data transform to rebalance a dataset by random over-sampling.
     *
     * This transform addresses class imbalance by duplicating samples from the
     * minority classes until they have as many samples as the majority class.
     * It operates on an entire dataset (features and labels) at once.
     *
     * USAGE:
     *   auto [resampled_features, resampled_labels] = ros.forward({features, labels});
     */
    class RandomOverSampling : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        RandomOverSampling();

        // NOTE: A more advanced version could take a sampling_strategy,
        // but the default 'auto' behavior is the most common use case.

        /**
         * @brief Resamples the provided dataset.
         * @param tensors An initializer list containing exactly two tensors:
         *                1. Features tensor (2D, shape [n_samples, n_features])
         *                2. Labels tensor (1D, shape [n_samples])
         * @return An std::any containing a std::pair of torch::Tensors:
         *         {resampled_features, resampled_labels}.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    };

} // namespace xt::transforms::general