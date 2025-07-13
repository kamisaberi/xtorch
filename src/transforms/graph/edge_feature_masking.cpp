#include "include/transforms/graph/edge_feature_masking.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph with edge features.
    torch::Tensor x = torch::randn({5, 8}); // 5 nodes, 8 features
    torch::Tensor edge_index = torch::tensor({
        {0, 1, 1, 2},
        {1, 0, 2, 1}
    }, torch::kLong);
    // Edge attributes: 4 edges, 10 features per edge
    torch::Tensor edge_attr = torch::ones({4, 10});

    std::cout << "Sum of edge features before masking: " << edge_attr.sum().item<float>() << std::endl;

    // 2. Create the transform to mask 30% of the edge feature dimensions.
    xt::transforms::graph::EdgeFeatureMasking masker(0.3);

    // 3. Apply the transform.
    auto result_any = masker.forward({x, edge_index, edge_attr});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_edge_attr = new_graph[2];

    // 4. Verify the output. The sum of edge features should be smaller.
    // We expect roughly 3 out of 10 feature columns to be zeroed out.
    // So the sum should be roughly 4 * 10 * (1 - 0.3) = 28.
    std::cout << "Sum of edge features after masking: " << new_edge_attr.sum().item<float>() << std::endl;
    std::cout << "Shape of new edge_attr: " << new_edge_attr.sizes() << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    EdgeFeatureMasking::EdgeFeatureMasking(double mask_rate, double p)
            : mask_rate_(mask_rate), p_(p) {

        if (mask_rate_ < 0.0 || mask_rate_ > 1.0) {
            throw std::invalid_argument("mask_rate must be in [0, 1].");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto EdgeFeatureMasking::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability Check ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 3) {
            throw std::invalid_argument("EdgeFeatureMasking expects 3 tensors: {node_features, edge_index, edge_attr}.");
        }

        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);
        torch::Tensor edge_attr = std::any_cast<torch::Tensor>(any_vec[2]);

        if (!x.defined() || !edge_index.defined() || !edge_attr.defined()) {
            throw std::invalid_argument("All input tensors must be defined.");
        }
        if (edge_index.size(1) != edge_attr.size(0)) {
            throw std::invalid_argument("Number of edges must match between edge_index and edge_attr.");
        }

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || mask_rate_ == 0.0) {
            return std::vector<torch::Tensor>{x, edge_index, edge_attr};
        }

        // --- 2. Mask Edge Features ---
        long num_edge_features = edge_attr.size(1);
        if (num_edge_features == 0) {
            return std::vector<torch::Tensor>{x, edge_index, edge_attr};
        }

        auto new_edge_attr = edge_attr.clone();

        // Create a mask for the feature dimensions (columns) to drop.
        auto drop_mask = torch::rand({num_edge_features}, edge_attr.options()) < mask_rate_;
        auto drop_indices = std::get<0>(torch::where(drop_mask));

        if (drop_indices.numel() > 0) {
            // Use advanced indexing to set the selected columns to zero.
            // The "..." selects all rows (all edges).
            new_edge_attr.index_put_({"...", drop_indices}, 0.0);
        }

        // --- 3. Return the new graph structure ---
        return std::vector<torch::Tensor>{x, edge_index, new_edge_attr};
    }

} // namespace xt::transforms::graph