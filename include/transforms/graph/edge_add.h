#pragma once

#include "../common.h"

namespace xt::transforms::graph {

    /**
     * @class EdgeAdd
     * @brief A graph transform that randomly adds new edges to the graph.
     *
     * This data augmentation technique can help regularize GNN models, especially on
     * sparse graphs, by creating new message-passing pathways. The transform avoids
     * adding self-loops or edges that already exist.
     *
     * @note For undirected graphs, this transform adds directed edges. The user is
     *       responsible for adding the reverse edges if symmetry is required.
     */
    class EdgeAdd : public xt::Module {
    public:
        /**
         * @brief Constructs the EdgeAdd transform.
         *
         * @param add_ratio The fraction of new edges to add, relative to the number
         *                  of existing edges. E.g., 0.1 means adding 10% new edges.
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit EdgeAdd(
                double add_ratio = 0.1,
                double p = 1.0);

        /**
         * @brief Executes the edge addition operation on a graph.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, num_node_features].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges], type long.
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the new graph `{node_features, new_edge_index}`. Node
         *         features are unchanged.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double add_ratio_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph