#pragma once

#include "../common.h"

namespace xt::transforms::graph {

    /**
     * @class FeatureDropout
     * @brief A graph transform that applies dropout to the node features.
     *
     * This is a standard regularization technique where a fraction of the elements in
     * the node feature tensor are randomly set to zero. During training, the remaining
     * elements are scaled up by `1 / (1 - drop_rate)` to maintain the same expected sum.
     */
    class FeatureDropout : public xt::Module {
    public:
        /**
         * @brief Constructs the FeatureDropout transform.
         *
         * @param drop_rate The probability of an element to be zeroed. Must be in [0, 1].
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit FeatureDropout(
                double drop_rate = 0.1,
                double p = 1.0);

        /**
         * @brief Executes the node feature dropout.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, num_features].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains `{new_node_features, edge_index}`. The edge_index is unchanged.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double drop_rate_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph