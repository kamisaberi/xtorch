#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class RandomInvert
     * @brief Inverts the colors of an image with a given probability.
     *
     * This transform creates a "photographic negative" effect by applying the
     * formula `output = 1.0 - input` to each pixel. The operation is defined
     * for float tensors in the [0, 1] range.
     * The operation is applied with a given probability `p`.
     */
    class RandomInvert : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies color inversion with a 50% probability.
         */
        RandomInvert();

        /**
         * @brief Constructs the RandomInvert transform.
         *
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomInvert(double p);

        /**
         * @brief Executes the random color inversion.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) with float values in the range [0, 1].
         * @return An std::any containing the resulting inverted torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image