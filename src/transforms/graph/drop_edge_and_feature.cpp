#include "include/transforms/graph/drop_edge_and_feature.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph.
    // Node features: 5 nodes, 16 features per node.
    torch::Tensor x = torch::randn({5, 16});
    // Edge index: 8 edges in a COO format.
    torch::Tensor edge_index = torch::tensor({
        {0, 1, 1, 2, 2, 3, 3, 4},
        {1, 0, 2, 1, 3, 2, 4, 3}
    }, torch::kLong);

    std::cout << "Original num nodes: " << x.size(0)
              << ", Original num edges: " << edge_index.size(1) << std::endl;
    std::cout << "Sum of node features before: " << x.sum().item<float>() << std::endl;

    // 2. Create the transform with high drop rates to see the effect clearly.
    xt::transforms::graph::DropEdgeAndFeature dropper(0.5, 0.3);

    // 3. Apply the transform.
    auto result_any = dropper.forward({x, edge_index});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = new_graph[0];
    torch::Tensor new_edge_index = new_graph[1];

    // 4. Verify the output.
    std::cout << "New num nodes: " << new_x.size(0)
              << ", New num edges: " << new_edge_index.size(1) << std::endl;
    std::cout << "Sum of node features after: " << new_x.sum().item<float>() << std::endl;
    // The number of edges should be smaller, and the sum of features should also be smaller.
    // The number of nodes remains the same.

    return 0;
}
*/

namespace xt::transforms::graph {

    DropEdgeAndFeature::DropEdgeAndFeature(double drop_edge_rate, double drop_feature_rate)
            : drop_edge_rate_(drop_edge_rate), drop_feature_rate_(drop_feature_rate) {

        if (drop_edge_rate_ < 0.0 || drop_edge_rate_ > 1.0) {
            throw std::invalid_argument("drop_edge_rate must be in [0, 1].");
        }
        if (drop_feature_rate_ < 0.0 || drop_feature_rate_ > 1.0) {
            throw std::invalid_argument("drop_feature_rate must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto DropEdgeAndFeature::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("DropEdgeAndFeature expects 2 tensors: {node_features, edge_index}.");
        }

        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!x.defined() || !edge_index.defined()) {
            throw std::invalid_argument("Input tensors must be defined.");
        }
        if (edge_index.size(0) != 2 || edge_index.dim() != 2) {
            throw std::invalid_argument("edge_index must have shape [2, num_edges].");
        }

        // --- 2. Drop Features ---
        torch::Tensor new_x = x.clone();
        if (drop_feature_rate_ > 0.0) {
            // This is the standard dropout implementation.
            double keep_prob = 1.0 - drop_feature_rate_;
            if (keep_prob > 0) {
                // Create a random binary mask and scale the remaining features.
                auto mask = torch::rand_like(x) < keep_prob;
                new_x = new_x * mask / keep_prob;
            } else {
                // If drop rate is 1.0, just zero out everything.
                new_x.zero_();
            }
        }

        // --- 3. Drop Edges ---
        torch::Tensor new_edge_index = edge_index.clone();
        if (drop_edge_rate_ > 0.0) {
            long num_edges = edge_index.size(1);
            if (num_edges > 0) {
                // Create a random mask for the edges to keep.
                auto edge_mask = torch::rand({num_edges}, x.options()) > drop_edge_rate_;

                // Select the columns from edge_index where the mask is true.
                auto keep_indices = std::get<0>(torch::where(edge_mask));
                new_edge_index = edge_index.index_select(/*dim=*/1, keep_indices);
            }
        }

        // --- 4. Return the new graph structure ---
        std::vector<torch::Tensor> result_graph = {new_x, new_edge_index};
        return result_any;
    }

} // namespace xt::transforms::graph