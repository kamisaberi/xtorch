#pragma once

#include "../common.h"


namespace xt::transforms::graph {

    /**
     * @class NodeFeatureMasking
     * @brief A graph transform that randomly masks (sets to zero) entire dimensions
     *        of the node feature tensor.
     *
     * This is a regularization technique for GNNs that helps prevent overfitting by
     * forcing the model to be robust to the absence of certain types of node
     * information across all nodes simultaneously.
     */
    class NodeFeatureMasking : public xt::Module {
    public:
        /**
         * @brief Constructs the NodeFeatureMasking transform.
         *
         * @param mask_rate The fraction of node feature dimensions to mask (set to zero).
         *                  Must be in [0, 1].
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit NodeFeatureMasking(
                double mask_rate = 0.1,
                double p = 1.0);

        /**
         * @brief Executes the node feature masking operation.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, num_features].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains `{new_node_features, edge_index}`. The edge_index is unchanged.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double mask_rate_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph
