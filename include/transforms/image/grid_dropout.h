#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class GridDropout
     * @brief An image transformation that randomly zeros out rectangular regions in a grid.
     *
     * This augmentation technique divides an image into a grid and randomly drops
     * a portion of the grid cells, forcing the model to learn from a more
     * distributed set of features.
     */
    class GridDropout : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses reasonable default parameters.
         */
        GridDropout();

        /**
         * @brief Constructs the GridDropout transform.
         * @param ratio The ratio of the grid cells to drop. Must be between 0.0 and 1.0.
         * @param unit_size_min The minimum size of a grid cell.
         * @param unit_size_max The maximum size of a grid cell. A random size in this
         *                      range will be chosen for each application.
         * @param holes_nb_min The minimum number of holes (if using ratio=0).
         * @param holes_nb_max The maximum number of holes (if using ratio=0).
         * @param fill_value The value to fill the dropped grid cells with.
         */
        GridDropout(
            float ratio = 0.5f,
            int unit_size_min = -1,
            int unit_size_max = -1,
            int holes_nb_min = 1,
            int holes_nb_max = -1,
            float fill_value = 0.0f
        );

        /**
         * @brief Executes the GridDropout operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor with grid
         *         regions erased. The shape and type remain the same.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float ratio_;
        int unit_size_min_;
        int unit_size_max_;
        int holes_nb_min_;
        int holes_nb_max_;
        float fill_value_;
    };

} // namespace xt::transforms::image