#pragma once

#include "../common.h"


#pragma once

#include "../common.h"
#include <random>

namespace xt::transforms::graph {

    /**
     * @class NodeDrop
     * @brief A graph transform that randomly drops a fraction of nodes and their
     *        incident edges from the graph.
     *
     * This is a strong structural regularization technique for GNNs. It forces the
     * model to learn robust representations that do not over-rely on any single
     * node or its direct connections.
     */
    class NodeDrop : public xt::Module {
    public:
        /**
         * @brief Constructs the NodeDrop transform.
         *
         * @param drop_rate The fraction of nodes to drop. Must be in [0, 1).
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit NodeDrop(
                double drop_rate = 0.1,
                double p = 1.0);

        /**
         * @brief Executes the node dropping operation.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, ...].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the new smaller graph `{new_node_features, new_edge_index}`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double drop_rate_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph