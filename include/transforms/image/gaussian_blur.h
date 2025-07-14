//TODO SHOULD CHANGE
#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class GaussianBlur
     * @brief Applies a Gaussian blur to an image with fixed parameters.
     *
     * This transform blurs an image using a Gaussian filter with a specified
     * kernel size and standard deviation (sigma). This is a deterministic operation.
     */
    class GaussianBlur : public xt::Module {
    public:
        /**
         * @brief Default constructor. Does nothing, parameters are required.
         */
        GaussianBlur();

        /**
         * @brief Constructs the GaussianBlur transform.
         *
         * @param kernel_size The size of the Gaussian kernel. Must be a positive, odd integer.
         * @param sigma A pair `{sigmaX, sigmaY}` for the standard deviation in each
         *              direction. If sigmaY is 0, it's set to be the same as sigmaX.
         * @param interpolation Not used in this transform, but kept for API consistency.
         */
        explicit GaussianBlur(
            int kernel_size,
            std::pair<double, double> sigma = {0.0, 0.0}
        );

        /**
         * @brief Executes the Gaussian blur.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting blurred torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int kernel_size_;
        std::pair<double, double> sigma_;
    };

} // namespace xt::transforms::image