#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class Spatter
     * @brief An image transformation that adds multiplicative speckle noise.
     *
     * This transform simulates effects like dust, rain, or sensor artifacts from
     * technologies like radar. It works by multiplying the image by a noise
     * field centered around 1.0.
     */
    class Spatter : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a mean of 0 and a standard deviation of 0.1.
         */
        Spatter();

        /**
         * @brief Constructs the Spatter transform.
         * @param mean The mean of the underlying Gaussian distribution used to
         *             generate the multiplicative noise. Usually 0.
         * @param sigma The standard deviation of the Gaussian distribution. This
         *              controls the intensity of the spatter effect.
         */
        Spatter(double mean, double sigma);

        /**
         * @brief Executes the spatter noise operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting noisy torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double mean_;
        double sigma_;
    };

} // namespace xt::transforms::image