#pragma once

#include "../common.h"



namespace xt::transforms::signal { // A new namespace for signal-specific transforms

    /**
     * @class AddNoise
     * @brief A signal transformation that adds Gaussian noise to a signal.
     *
     * This is a fundamental data augmentation technique for time-series or audio
     * data, used to improve a model's robustness to noisy inputs.
     */
    class AddNoise : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a moderate noise amplitude (sigma) of 0.01.
         */
        AddNoise();

        /**
         * @brief Constructs the AddNoise transform.
         * @param min_amplitude The minimum amplitude (standard deviation) of the noise.
         * @param max_amplitude The maximum amplitude (standard deviation) of the noise.
         *                      A random amplitude in this range will be chosen for each call.
         * @param p The probability of applying noise to the signal.
         */
        AddNoise(float min_amplitude, float max_amplitude, float p = 0.5f);

        /**
         * @brief Executes the noise addition operation.
         * @param tensors An initializer list expected to contain a single signal
         *                tensor (e.g., shape [num_samples] or [num_channels, num_samples]).
         * @return An std::any containing the resulting noisy torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float min_amplitude_;
        float max_amplitude_;
        float p_;
    };

} // namespace xt::transforms::signal