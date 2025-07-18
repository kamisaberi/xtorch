#include <transforms/graph/node_feature_masking.h>


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph.
    torch::Tensor x = torch::ones({5, 10}); // 5 nodes, 10 features
    torch::Tensor edge_index = torch::tensor({{0, 1}, {1, 0}}, torch::kLong);

    std::cout << "Sum of node features before masking: " << x.sum().item<float>() << std::endl;

    // 2. Create the transform to mask 40% of the node feature dimensions.
    xt::transforms::graph::NodeFeatureMasking masker(0.4);

    // 3. Apply the transform.
    auto result_any = masker.forward({x, edge_index});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = new_graph[0];

    // 4. Verify the output. The sum of features should be smaller.
    // We expect 4 out of 10 feature columns to be zeroed out.
    // So the sum should be 5 * 10 * (1 - 0.4) = 30.
    std::cout << "Sum of node features after masking: " << new_x.sum().item<float>() << std::endl;
    std::cout << "Shape of new node features: " << new_x.sizes() << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    NodeFeatureMasking::NodeFeatureMasking(double mask_rate, double p)
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

    auto NodeFeatureMasking::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability Check ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("NodeFeatureMasking expects 2 tensors: {node_features, edge_index}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || mask_rate_ == 0.0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        // --- 2. Mask Node Features ---
        long num_features = x.size(1);
        if (num_features == 0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        auto new_x = x.clone();

        // Create a mask for the feature dimensions (columns) to drop.
        auto drop_mask = torch::rand({num_features}, x.options()) < mask_rate_;
        // auto drop_indices = std::get<0>(torch::where(drop_mask));
        auto drop_indices = torch::where(drop_mask)[0];


        if (drop_indices.numel() > 0) {
            // Use advanced indexing to set the selected columns to zero for all nodes.
            // The "..." selects all rows (all nodes).
            new_x.index_put_({"...", drop_indices}, 0.0);
        }

        // --- 3. Return the new graph structure ---
        return std::vector<torch::Tensor>{new_x, edge_index};
    }

} // namespace xt::transforms::graph