#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class RandomAutoContrast
     * @brief Automatically adjusts the contrast of an image by stretching the pixel
     *        intensities to span a desired range.
     *
     * This transform automatically adjusts the contrast of an image based on its
     * pixel distribution. It works by clipping the bottom-most and top-most
     * `cutoff` percent of the pixel values (per channel) and then stretching the
     * remaining values to the full 8-bit range.
     *
     * The transform is applied with a given probability `p`.
     */
    class RandomAutoContrast : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies auto-contrast with 50% probability
         *        and a cutoff of 0.0 (meaning no clipping).
         */
        RandomAutoContrast();

        /**
         * @brief Constructs the RandomAutoContrast transform.
         *
         * @param cutoff The percentage of pixels to cut off from each channel's
         *               distribution. Must be between 0.0 and 0.5.
         *               A cutoff of 0.0 means no clipping, and the contrast
         *               will be stretched to the full range.
         * @param p The probability of applying the transform. Must be in [0, 1].
         */
        explicit RandomAutoContrast(
            double cutoff = 0.0,
            double p = 0.5
        );

        /**
         * @brief Executes the auto-contrast adjustment.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W). The tensor is expected to be in the
         *                [0, 1] float range.
         * @return An std::any containing the resulting torch::Tensor with adjusted
         *         contrast. The image may be unchanged if the probability check fails
         *         or if the pixel distribution does not benefit from contrast adjustment
         *         with the given cutoff.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double cutoff_;
        double p_;

        std::mt19937 gen_;
    };

} // namespace xt::transforms::image