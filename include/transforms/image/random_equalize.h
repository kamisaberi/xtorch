#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomEqualize
     * @brief Applies histogram equalization to an image with a given probability.
     *
     * This transform improves the contrast in an image by "spreading out" the
     * most frequent intensity values. It works by applying a non-linear mapping
     * to the pixel intensities which results in a flat or nearly flat histogram.
     * The operation is applied with a given probability `p`.
     */
    class RandomEqualize : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies histogram equalization with a 50% probability.
         */
        RandomEqualize();

        /**
         * @brief Constructs the RandomEqualize transform.
         *
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomEqualize(double p);

        /**
         * @brief Executes the histogram equalization.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting equalized torch::Tensor.
         *         The image may be unchanged if the probability check fails.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image