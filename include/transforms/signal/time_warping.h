#pragma once

#include "../common.h"



namespace xt::transforms::signal {

    /**
     * @class TimeWarping
     * @brief A transform that non-linearly warps the time axis of a waveform.
     *
     * This data augmentation technique simulates variations in timing and cadence by
     * picking an anchor point in time, shifting it, and then stretching/compressing
     * the surrounding signal to fit the original duration. The overall length of
     * the waveform does not change.
     */
    class TimeWarping : public xt::Module {
    public:
        /**
         * @brief Constructs the TimeWarping transform.
         *
         * @param max_time_warp_percent The maximum percentage of the total duration
         *                              that the central anchor point can be shifted.
         *                              For example, 0.05 means the anchor can move
         *                              up to 5% of the total length forward or backward.
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit TimeWarping(
                double max_time_warp_percent = 0.05,
                double p = 1.0);

        /**
         * @brief Executes the time warping operation.
         *
         * @param tensors An initializer list expected to contain a single 1D audio
         *                tensor (waveform).
         * @return An std::any containing the time-warped waveform as a
         *         torch::Tensor, with the same length as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double max_time_warp_percent_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::signal