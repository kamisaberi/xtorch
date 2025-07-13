#pragma once

#include "../common.h"


namespace xt::transforms::signal {

    /**
     * @class SlidingWindowCMN
     * @brief A transform to apply Cepstral Mean and (optionally) Variance Normalization
     *        over a sliding window.
     *
     * This is a common feature normalization technique in speech recognition that
     * makes the model more robust to different recording channels and speakers.
     * It normalizes each frame based on the statistics of its neighboring frames.
     */
    class SlidingWindowCMN : public xt::Module {
    public:
        /**
         * @brief Constructs the SlidingWindowCMN transform.
         *
         * @param cmn_window The size of the sliding window for CMN. The window is
         *                   centered, so it includes `(cmn_window - 1) / 2` frames
         *                   on each side of the current frame.
         * @param min_cmn_window The minimum number of frames to use for CMN at the
         *                       boundaries of the utterance. Defaults to `cmn_window`.
         * @param normalize_mean Whether to subtract the window's mean. Defaults to true.
         * @param normalize_variance Whether to divide by the window's standard deviation.
         *                           Defaults to false.
         * @param center Whether to center the window around the current frame.
         *               Defaults to true. If false, the window is causal (looks only
         *               at past frames).
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit SlidingWindowCMN(
                int cmn_window = 600,
                int min_cmn_window = 100,
                bool normalize_mean = true,
                bool normalize_variance = false,
                bool center = true,
                double p = 1.0);

        /**
         * @brief Executes the sliding window normalization.
         *
         * @param tensors An initializer list expected to contain a single feature tensor
         *                (e.g., MFCCs) of shape (num_features, num_frames).
         * @return An std::any containing the resulting normalized feature tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int cmn_window_;
        int min_cmn_window_;
        bool normalize_mean_;
        bool normalize_variance_;
        bool center_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::signal