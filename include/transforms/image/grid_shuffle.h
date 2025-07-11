#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class GridShuffle
     * @brief An image transformation that shuffles grid cells within an image.
     *
     * This augmentation divides the image into a grid of non-overlapping cells,
     * randomly shuffles the locations of these cells, and reassembles the image.
     * This forces the model to learn features that are less sensitive to the
     * absolute spatial position of objects.
     */
    class GridShuffle : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a 4x4 grid.
         */
        GridShuffle();

        /**
         * @brief Constructs the GridShuffle transform.
         * @param grid_size The number of cells along each axis (e.g., 4 for a 4x4 grid).
         */
        explicit GridShuffle(int grid_size);

        /**
         * @brief Executes the grid shuffling operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting shuffled torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int grid_size_;
    };

} // namespace xt::transforms::image