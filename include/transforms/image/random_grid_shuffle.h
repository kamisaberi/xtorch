#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class RandomGridShuffle
     * @brief Divides an image into a grid and randomly shuffles the grid cells.
     *
     * This transform encourages the model to learn features that are invariant
     * to the spatial arrangement of local patterns. It divides the image into
     * a grid of `grid_size` x `grid_size` cells and then permutes these cells.
     * The operation is applied with a given probability `p`.
     */
    class RandomGridShuffle : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates a 4x4 grid shuffle with 50% probability.
         */
        RandomGridShuffle();

        /**
         * @brief Constructs the RandomGridShuffle transform.
         *
         * @param grid_size The number of cells along each dimension. A value of 4
         *                  creates a 4x4 grid of cells.
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomGridShuffle(
            int grid_size,
            double p = 0.5
        );

        /**
         * @brief Executes the random grid shuffle.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting shuffled torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int grid_size_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image