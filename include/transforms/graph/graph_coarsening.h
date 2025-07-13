#pragma once

#include "../common.h"

namespace xt::transforms::graph {

    /**
     * @class GraphCoarsening
     * @brief A graph transform that reduces the number of nodes in a graph by
     *        merging them, while attempting to preserve the graph's structure.
     *
     * This transform uses a greedy matching algorithm to cluster nodes and then
     * pools their features and reconstructs the edge connectivity for the new,
     * smaller graph. It is a key component in graph pooling and multi-level GNNs.
     */
    class GraphCoarsening : public xt::Module {
    public:
        /**
         * @brief Constructs the GraphCoarsening transform.
         *
         * @param coarsening_ratio The desired fraction of nodes to have in the
         *                         coarsened graph. E.g., 0.5 means the new graph
         *                         will have approximately half the nodes of the original.
         *                         Must be in (0, 1].
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit GraphCoarsening(
                double coarsening_ratio = 0.5,
                double p = 1.0);

        /**
         * @brief Executes the graph coarsening operation.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, num_features].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the new coarsened graph `{new_node_features, new_edge_index}`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double coarsening_ratio_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph