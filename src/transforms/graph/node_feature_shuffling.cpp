#include <transforms/graph/node_feature_shuffling.h>


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph with easily identifiable features.
    // The feature of node `i` is simply `i`.
    torch::Tensor x = torch::arange(0, 5, torch::kFloat32).view({-1, 1});
    torch::Tensor edge_index = torch::tensor({
        {0, 1, 2, 3, 4},
        {1, 2, 3, 4, 0} // A 5-node cycle
    }, torch::kLong);

    std::cout << "Original node features:\n" << x << std::endl;

    // 2. Create the NodeFeatureShuffling transform.
    xt::transforms::graph::NodeFeatureShuffling shuffler;

    // 3. Apply the transform.
    auto result_any = shuffler.forward({x, edge_index});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = new_graph[0];
    torch::Tensor new_edge_index = new_graph[1];

    // 4. Verify the output. The features should be a permutation of the original.
    // The edge_index remains the same.
    std::cout << "\nShuffled node features:\n" << new_x << std::endl;
    std::cout << "Edge index (unchanged):\n" << new_edge_index << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    NodeFeatureShuffling::NodeFeatureShuffling(double p)
            : p_(p) {

        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto NodeFeatureShuffling::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("NodeFeatureShuffling expects 2 tensors: {node_features, edge_index}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        long num_nodes = x.size(0);
        if (num_nodes < 2) {
            // Cannot shuffle 0 or 1 items.
            return std::vector<torch::Tensor>{x, edge_index};
        }

        // --- 2. Create a Random Permutation of Node Indices ---
        std::vector<long> perm(num_nodes);
        std::iota(perm.begin(), perm.end(), 0); // Fills vector with 0, 1, 2, ...
        std::shuffle(perm.begin(), perm.end(), random_engine_);

        // --- 3. Apply the Permutation to the Node Feature Matrix ---
        // `torch::index_select` is the efficient, vectorized way to do this.
        auto perm_tensor = torch::tensor(perm, torch::kLong).to(x.device());
        torch::Tensor new_x = x.index_select(/*dim=*/0, perm_tensor);

        // --- 4. Return the new graph structure ---
        // The edge_index is unchanged, as the graph's structure is preserved.
        return std::vector<torch::Tensor>{new_x, edge_index};
    }

} // namespace xt::transforms::graph