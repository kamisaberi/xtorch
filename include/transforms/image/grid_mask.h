#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class GridMask
     * @brief An image transformation that applies the GridMask data augmentation.
     *
     * This technique erases a structured grid of smaller squares from the image,
     * forcing the model to learn more robust and context-aware features.
     * The grid can be randomly rotated and offset for variability.
     */
    class GridMask : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses reasonable default parameters.
         */
        GridMask();

        /**
         * @brief Constructs the GridMask transform.
         * @param d_min The minimum size of the grid cycle (d).
         * @param d_max The maximum size of the grid cycle (d). A random d in this
         *              range is chosen for each application.
         * @param ratio The ratio that determines the size of the dropped square
         *              relative to the grid cycle size.
         * @param rotate The maximum angle in degrees to randomly rotate the grid.
         * @param p The probability of applying the transform.
         */
        GridMask(
            int d_min,
            int d_max,
            double ratio,
            double rotate = 0.0,
            double p = 0.5
        );

        /**
         * @brief Executes the GridMask operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor with the grid
         *         mask applied. The shape and type remain the same.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int d_min_;
        int d_max_;
        double ratio_;
        double rotate_;
        double p_;
    };

} // namespace xt::transforms::image