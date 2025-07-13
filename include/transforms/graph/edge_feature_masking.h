#pragma once

#include "../common.h"



namespace xt::transforms::graph {

    /**
     * @class EdgeFeatureMasking
     * @brief A graph transform that randomly masks (sets to zero) entire dimensions
     *        of the edge feature tensor.
     *
     * This is a regularization technique for GNNs that operate on graphs with edge
     * attributes. It helps prevent overfitting by forcing the model to be robust
     * to the absence of certain types of edge information.
     */
    class EdgeFeatureMasking : public xt::Module {
    public:
        /**
         * @brief Constructs the EdgeFeatureMasking transform.
         *
         * @param mask_rate The fraction of edge feature dimensions to mask (set to zero).
         *                  Must be in [0, 1].
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit EdgeFeatureMasking(
                double mask_rate = 0.1,
                double p = 1.0);

        /**
         * @brief Executes the edge feature masking operation.
         *
         * @param tensors An initializer list expected to contain three tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, num_node_features].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         *                3. `edge_attr` (torch::Tensor): Shape [num_edges, num_edge_features].
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the new graph `{node_features, edge_index, new_edge_attr}`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double mask_rate_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph