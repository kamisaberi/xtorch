#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomGrayscale
     * @brief Converts an image to grayscale with a given probability.
     *
     * This transform converts an RGB image to its single-channel luminance
     * representation and then duplicates that channel three times to produce
     * a grayscale image with the same number of channels as the input.
     * The operation is applied with a given probability `p`.
     */
    class RandomGrayscale : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies grayscale conversion with a 10% probability.
         */
        RandomGrayscale();

        /**
         * @brief Constructs the RandomGrayscale transform.
         *
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomGrayscale(double p);

        /**
         * @brief Executes the random grayscale conversion.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor. If the input
         *         is already grayscale or the probability check fails, the original
         *         tensor is returned.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image