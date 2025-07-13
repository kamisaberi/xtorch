#pragma once

#include "../common.h"


namespace xt::transforms::graph {

    /**
     * @class RandomWalkSubgraph
     * @brief A graph transform that extracts a subgraph by performing random walks.
     *
     * This technique is commonly used for sampling from large graphs to create
     * mini-batches for training GNNs. It creates a localized subgraph around a
     * set of starting nodes.
     */
    class RandomWalkSubgraph : public xt::Module {
    public:
        /**
         * @brief Constructs the RandomWalkSubgraph transform.
         *
         * @param num_start_nodes The number of nodes to use as starting points for
         *                        the random walks.
         * @param walk_length The length of each random walk (number of steps).
         */
        explicit RandomWalkSubgraph(
                int num_start_nodes,
                int walk_length);

        /**
         * @brief Executes the random walk sampling.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, ...].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the new subgraph `{subgraph_node_features, subgraph_edge_index}`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int num_start_nodes_;
        int walk_length_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph