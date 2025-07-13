#pragma once

#include "../common.h"

namespace xt::transforms::signal {

    /**
     * @class Resample
     * @brief A transform that resamples a waveform from one sample rate to another.
     *
     * This transform uses the high-quality `libsamplerate` (Secret Rabbit Code) library
     * to perform anti-aliased resampling, which is crucial for preventing artifacts
     * when changing the sample rate of an audio signal.
     */
    class Resample : public xt::Module {
    public:
        /**
         * @brief Constructs the Resample transform.
         *
         * @param orig_freq The original sample rate of the input waveform in Hz.
         * @param new_freq The target sample rate to resample to in Hz.
         * @param quality The resampling quality (maps to libsamplerate converters).
         *                0: SINC_BEST_QUALITY (default)
         *                1: SINC_MEDIUM_QUALITY
         *                2: SINC_FASTEST
         *                3: ZERO_ORDER_HOLD
         *                4: LINEAR
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit Resample(
                int orig_freq,
                int new_freq,
                int quality = 0,
                double p = 1.0);

        /**
         * @brief Executes the resampling operation.
         *
         * @param tensors An initializer list expected to contain a single 1D (waveform)
         *                or 2D (batch, waveform) audio tensor.
         * @return An std::any containing the resulting resampled waveform as a
         *         torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int orig_freq_;
        int new_freq_;
        int quality_;
        double p_;
        bool is_identity_; // True if orig_freq == new_freq
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::signal