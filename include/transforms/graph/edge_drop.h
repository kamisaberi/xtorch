#pragma once

#include "../common.h"

namespace xt::transforms::graph {

    /**
     * @class EdgeDrop
     * @brief A graph transform that randomly drops edges from the graph's connectivity.
     *
     * This is a common regularization technique for Graph Neural Networks (GNNs),
     * analogous to Dropout. It helps prevent overfitting by forcing the model to
     * learn from a sparser version of the graph in each training step.
     */
    class EdgeDrop : public xt::Module {
    public:
        /**
         * @brief Constructs the EdgeDrop transform.
         *
         * @param drop_rate The fraction of edges to drop. Must be in [0, 1].
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit EdgeDrop(
                double drop_rate = 0.1,
                double p = 1.0);

        /**
         * @brief Executes the edge dropping operation on a graph.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, num_node_features].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges], type long.
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains `{node_features, new_edge_index}`. Node features are unchanged.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double drop_rate_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph