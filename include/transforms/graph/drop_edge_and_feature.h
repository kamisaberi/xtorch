#pragma once

#include "../common.h"

namespace xt::transforms::graph {

    /**
     * @class DropEdgeAndFeature
     * @brief A graph transform that randomly drops edges and node features.
     *
     * This is a common regularization technique for training Graph Neural Networks (GNNs).
     * It helps prevent the model from overfitting to specific structural patterns or
     * node features.
     *
     * - **DropEdge**: Randomly removes a fraction of edges from the graph's connectivity.
     * - **DropFeature**: Randomly sets a fraction of node features to zero.
     */
    class DropEdgeAndFeature : public xt::Module {
    public:
        /**
         * @brief Constructs the DropEdgeAndFeature transform.
         *
         * @param drop_edge_rate The fraction of edges to drop. Must be in [0, 1].
         * @param drop_feature_rate The fraction of node features to set to zero.
         *                          Must be in [0, 1].
         */
        explicit DropEdgeAndFeature(
                double drop_edge_rate = 0.1,
                double drop_feature_rate = 0.1);

        /**
         * @brief Executes the dropping operation on a graph.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, num_node_features].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges], type long.
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the new `{node_features, edge_index}`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double drop_edge_rate_;
        double drop_feature_rate_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph