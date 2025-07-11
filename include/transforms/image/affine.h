#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class Affine
     * @brief An image transformation that applies a 2D affine transformation.
     *
     * This transform can combine rotation, translation, scaling, and shearing
     * into a single operation. It is a powerful tool for data augmentation.
     * This implementation uses OpenCV for the underlying computation.
     */
    class Affine : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates an identity transform (does nothing).
         */
        Affine();

        /**
         * @brief Constructs the Affine transform with specified parameters.
         * @param degrees The angle of rotation in degrees.
         * @param translate A vector of two doubles {tx, ty} representing the
         *                  translation as a fraction of image size. E.g., {0.1, 0.1}
         *                  shifts the image by 10% of its width and height.
         * @param scale The scaling factor. 1.0 means no scaling.
         * @param shear The shear angle in degrees.
         */
        Affine(double degrees, const std::vector<double>& translate, double scale, double shear);

        /**
         * @brief Executes the affine transformation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting transformed torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double degrees_;
        std::vector<double> translate_;
        double scale_;
        double shear_;
    };

} // namespace xt::transforms::image