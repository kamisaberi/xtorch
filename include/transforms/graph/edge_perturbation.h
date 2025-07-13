#pragma once

#include "../common.h"


namespace xt::transforms::graph {

    /**
     * @class EdgePerturbation
     * @brief A graph transform that simultaneously drops existing edges and adds
     *        new, non-existent ones.
     *
     * This is a powerful data augmentation technique that combines EdgeDrop and
     * EdgeAdd to holistically modify the graph's topology, making GNN models
     * more robust to structural noise.
     */
    class EdgePerturbation : public xt::Module {
    public:
        /**
         * @brief Constructs the EdgePerturbation transform.
         *
         * @param add_ratio The fraction of new edges to add, relative to the number
         *                  of original edges.
         * @param drop_ratio The fraction of existing edges to drop.
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit EdgePerturbation(
                double add_ratio = 0.1,
                double drop_ratio = 0.1,
                double p = 1.0);

        /**
         * @brief Executes the edge perturbation operation.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, ...].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the new graph `{node_features, new_edge_index}`. Node
         *         features are unchanged.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double add_ratio_;
        double drop_ratio_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::graph