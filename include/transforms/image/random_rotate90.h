#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class RandomRotate90
     * @brief Rotates an image by 0, 90, 180, or 270 degrees with a given probability.
     *
     * This transform randomly selects a 90-degree increment rotation and applies it
     * to the image. These rotations are fast and lossless as they do not require
     * interpolation.
     * The operation is applied with a given probability `p`.
     */
    class RandomRotate90 : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies a random 90-degree rotation with 50% probability.
         */
        RandomRotate90();

        /**
         * @brief Constructs the RandomRotate90 transform.
         *
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomRotate90(double p = 0.5);

        /**
         * @brief Executes the random 90-degree rotation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting rotated torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image