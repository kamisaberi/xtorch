#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class PiecewiseAffine
     * @brief An image transformation that applies localized non-rigid distortions.
     *
     * This transform divides an image into a grid of triangles and applies a
     * separate affine transformation to each triangle, creating a complex "wavy"
     * or "liquify" effect. It is a very strong geometric augmentation.
     * This implementation uses a combination of OpenCV functionalities.
     */
    class PiecewiseAffine : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses reasonable default parameters.
         */
        PiecewiseAffine();

        /**
         * @brief Constructs the PiecewiseAffine transform.
         * @param scale A factor controlling the intensity of the random displacement
         *              of grid points, as a fraction of image size.
         * @param nb_rows The number of rows of grid points.
         * @param nb_cols The number of columns of grid points.
         * @param p The probability of applying the transform.
         */
        PiecewiseAffine(
            float scale = 0.05f,
            int nb_rows = 4,
            int nb_cols = 4,
            float p = 0.5f
        );

        /**
         * @brief Executes the piecewise affine distortion.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting distorted torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float scale_;
        int nb_rows_;
        int nb_cols_;
        float p_;
    };

} // namespace xt::transforms::image