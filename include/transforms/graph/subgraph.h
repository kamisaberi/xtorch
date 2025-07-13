#pragma once

#include "../common.h"



namespace xt::transforms::graph {

    /**
     * @class Subgraph
     * @brief A graph transform that extracts a subgraph based on a given set of node indices.
     *
     * This utility takes a graph and a list of nodes to keep. It returns a new graph
     * containing only those nodes and the edges that exist between them. The nodes
     * and edges in the resulting subgraph are re-indexed to be contiguous.
     */
    class Subgraph : public xt::Module {
    public:
        /**
         * @brief Constructs the Subgraph transform. This is a parameter-free transform.
         */
        explicit Subgraph();

        /**
         * @brief Executes the subgraph extraction.
         *
         * @param tensors An initializer list expected to contain three tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, ...].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         *                3. `nodes_to_keep` (torch::Tensor): A 1D tensor of type long
         *                   containing the indices of the nodes to include in the subgraph.
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the new subgraph `{subgraph_node_features, subgraph_edge_index}`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    };

} // namespace xt::transforms::graph