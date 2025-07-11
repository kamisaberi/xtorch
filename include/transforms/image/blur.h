#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class Blur
     * @brief An image transformation that applies a Gaussian blur.
     *
     * This transform smooths an image by convolving it with a Gaussian kernel.
     * It is a common data augmentation technique and pre-processing step.
     * This implementation uses OpenCV for the underlying computation.
     */
    class Blur : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a 3x3 kernel and auto-calculated sigma.
         */
        Blur();

        /**
         * @brief Constructs the Gaussian Blur transform with specific parameters.
         * @param kernel_size A vector of two odd integers {height, width} defining the
         *                    size of the Gaussian kernel.
         * @param sigma A vector of two doubles {sigmaX, sigmaY} for the Gaussian kernel
         *              standard deviation. If {0, 0}, sigma is calculated from kernel_size.
         */
        Blur(std::vector<int64_t> kernel_size, std::vector<double> sigma);

        /**
         * @brief Executes the Gaussian blur operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting blurred torch::Tensor with the
         *         same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::vector<int64_t> kernel_size_;
        std::vector<double> sigma_;
    };

} // namespace xt::transforms::image