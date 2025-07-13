#pragma once

#include "../common.h"

namespace xt::transforms::graph {

    /**
     * @class NodeMixUp
     * @brief A graph transform that creates new node features by interpolating between
     *        pairs of nodes within the same graph.
     *
     * This data augmentation technique generates "virtual" nodes by taking a weighted
     * average of features from randomly paired nodes. The interpolation weight `lambda`
     * is sampled from a Beta distribution. This encourages the model to learn
     * smoother representations.
     */
    class NodeMixUp : public xt::Module {
    public:
        /**
         * @brief Constructs the NodeMixUp transform.
         *
         * @param alpha The concentration parameter for the Beta distribution. Higher
         *              alpha pushes the interpolation weight `lambda` towards 0.5.
         * @param mixup_ratio The fraction of nodes to select for mixing. For example,
         *                    0.2 means 20% of the nodes will be paired up and mixed.
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit NodeMixUp(
                double alpha = 0.2,
                double mixup_ratio = 0.2,
                double p = 1.0);

        /**
         * @brief Executes the node mixup operation.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, ...].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains `{new_node_features, edge_index}`. The edge_index is unchanged.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double alpha_;
        double mixup_ratio_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph