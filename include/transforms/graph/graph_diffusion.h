#pragma once

#include "../common.h"


namespace xt::transforms::graph {

    /**
     * @class GraphDiffusion
     * @brief A graph transform that applies a diffusion process to the graph's
     *        features and optionally its structure.
     *
     * This transform uses the graph Laplacian to simulate a heat diffusion process,
     * which acts as a low-pass filter. It can be used to smooth node features
     * and infer new, structurally-plausible edges. The diffusion operator is
     * approximated using a Taylor series expansion for efficiency.
     */
    class GraphDiffusion : public xt::Module {
    public:
        /**
         * @brief Constructs the GraphDiffusion transform.
         *
         * @param beta The diffusion time/strength. Higher values lead to more smoothing.
         * @param add_self_loops Whether to add self-loops to the graph before diffusion.
         *                       This is highly recommended. Defaults to true.
         * @param k The number of terms to use in the Taylor series approximation of the
         *          matrix exponential. Higher values are more accurate but slower.
         *          Defaults to 8.
         * @param add_new_edges_threshold If > 0, new edges will be added where the
         *                                diffused connectivity value is above this
         *                                threshold. Set to 0 to disable adding edges.
         *                                Defaults to 0.
         */
        explicit GraphDiffusion(
                double beta = 1.0,
                bool add_self_loops = true,
                int k = 8,
                double add_new_edges_threshold = 0.0);

        /**
         * @brief Executes the graph diffusion process.
         *
         * @param tensors An initializer list expected to contain two tensors:
         *                1. `node_features` (torch::Tensor): Shape [num_nodes, num_features].
         *                2. `edge_index` (torch::Tensor): Shape [2, num_edges].
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the new diffused graph `{new_node_features, new_edge_index}`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double beta_;
        bool add_self_loops_;
        int k_;
        double add_new_edges_threshold_;
    };

} // namespace xt::transforms::graph