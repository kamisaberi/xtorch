#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class FancyPCA
     * @brief An image transformation that applies PCA-based color augmentation.
     *
     * This technique, made famous by the AlexNet paper, alters the RGB channels of an
     * image based on the principal components of color variation found in a large
     * dataset (like ImageNet). It adds color jitter that is consistent with natural
     * image statistics.
     *
     * This transform requires pre-computed eigenvectors and eigenvalues from the
     * target dataset.
     */
    class FancyPCA : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates an uninitialized (and unusable) transform.
         */
        FancyPCA();

        /**
         * @brief Constructs the FancyPCA transform.
         * @param eigenvectors A 3x3 torch::Tensor where each row is a principal
         *                     component of the RGB color space.
         * @param eigenvalues A 1D torch::Tensor of size 3 containing the standard
         *                    deviation (square root of eigenvalue) for each component.
         * @param alpha A random strength factor, drawn uniformly from [0, alpha_std].
         *              If 0, this parameter is ignored.
         */
        explicit FancyPCA(torch::Tensor eigenvectors, torch::Tensor eigenvalues, double alpha_std = 0.1);

        /**
         * @brief Executes the PCA color augmentation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) with C=3.
         * @return An std::any containing the resulting color-jittered torch::Tensor
         *         with the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        torch::Tensor eigenvectors_;
        torch::Tensor eigenvalues_;
        double alpha_std_;
    };

} // namespace xt::transforms::image