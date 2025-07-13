#pragma once

#include "../common.h"

namespace xt::transforms::signal {

    /**
     * @class Vol
     * @brief A transform that adjusts the volume of a waveform.
     *
     * This data augmentation technique multiplies the waveform by a random scaling
     * factor, chosen from a specified decibel (dB) range. This helps make models
     * robust to different recording levels. The output is clamped to [-1.0, 1.0]
     * to prevent clipping.
     */
    class Vol : public xt::Module {
    public:
        /**
         * @brief Constructs the Vol (Volume) transform.
         *
         * @param min_db The minimum volume change in decibels. Negative values decrease
         *               volume, positive values increase it.
         * @param max_db The maximum volume change in decibels.
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit Vol(
                double min_db,
                double max_db,
                double p = 1.0);

        /**
         * @brief Executes the volume adjustment operation.
         *
         * @param tensors An initializer list expected to contain a single 1D audio
         *                tensor (waveform).
         * @return An std::any containing the volume-adjusted waveform as a
         *         torch::Tensor, with the same length as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double min_db_;
        double max_db_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::signal