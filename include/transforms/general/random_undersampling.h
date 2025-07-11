#pragma once
#include "../common.h"




namespace xt::transforms::general {

    /**
     * @class RandomUnderSampling
     * @brief A data transform to rebalance a dataset by random under-sampling.
     *
     * This transform addresses class imbalance by randomly removing samples from the
     * majority classes until they have as many samples as the minority class.
     * It operates on an entire dataset (features and labels) at once.
     *
     * USAGE:
     *   auto [resampled_features, resampled_labels] = rus.forward({features, labels});
     */
    class RandomUnderSampling : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        RandomUnderSampling();

        // NOTE: A more advanced version could take a sampling_strategy,
        // but the default 'auto' behavior is the most common use case.

        /**
         * @brief Resamples the provided dataset by under-sampling the majority classes.
         * @param tensors An initializer list containing exactly two tensors:
         *                1. Features tensor (2D, shape [n_samples, n_features])
         *                2. Labels tensor (1D, shape [n_samples])
         * @return An std::any containing a std::pair of torch::Tensors:
         *         {resampled_features, resampled_labels}.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    };

} // namespace xt::transforms::general