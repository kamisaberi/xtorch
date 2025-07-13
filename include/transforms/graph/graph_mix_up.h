#pragma once

#include "../common.h"



namespace xt::transforms::graph {

    /**
     * @class GraphMixUp
     * @brief A graph transform that creates a new graph by interpolating between two
     *        graphs from a dataset.
     *
     * This data augmentation technique generates synthetic training examples by taking
     * a weighted average of two graphs. The node features are interpolated, and the
     * edge sets are combined. The interpolation weight is sampled from a Beta
     * distribution.
     *
     * @note This transform expects to be applied to a pair of graphs.
     */
    class GraphMixUp : public xt::Module {
    public:
        /**
         * @brief Constructs the GraphMixUp transform.
         *
         * @param alpha The concentration parameter for the Beta distribution, which
         *              controls the strength of the interpolation. Higher alpha pushes
         *              the interpolation weight `lambda` towards 0.5. Defaults to 0.2.
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit GraphMixUp(
                double alpha = 0.2,
                double p = 1.0);

        /**
         * @brief Executes the GraphMixUp operation.
         *
         * @param tensors An initializer list expected to contain four tensors from two graphs:
         *                1. `x1` (torch::Tensor): Node features of graph 1.
         *                2. `edge_index1` (torch::Tensor): Edge index of graph 1.
         *                3. `x2` (torch::Tensor): Node features of graph 2.
         *                4. `edge_index2` (torch::Tensor): Edge index of graph 2.
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the new mixed graph `{new_x, new_edge_index}`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double alpha_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph