#pragma once

#include "../common.h"


namespace xt::transforms::graph {

    /**
     * @class NodeFeatureShuffling
     * @brief A graph transform that randomly shuffles the feature vectors among the nodes.
     *
     * This data augmentation technique decouples a node's features from its structural
     * position in the graph. Each node keeps its original connectivity, but its
     * feature vector is randomly swapped with that of another node. This encourages
     * the model to learn more generalizable structural and feature-based patterns.
     */
    class NodeFeatureShuffling : public xt::Module {
    public:
        /**
         * @brief Constructs the NodeFeatureShuffling transform.
         *
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit NodeFeatureShuffling(double p = 1.0);

        /**
         * @brief Executes the node feature shuffling operation.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, ...].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains `{new_node_features, edge_index}`. The edge_index is unchanged.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph