#include <transforms/graph/edge_drop.h>



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

    // 2. Create the transform to drop 25% of the edges.
    xt::transforms::graph::EdgeDrop edge_dropper(0.25);

    // 3. Apply the transform.
    auto result_any = edge_dropper.forward({x, edge_index});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = new_graph[0];
    torch::Tensor new_edge_index = new_graph[1];

    // 4. Verify the output. The number of edges should be smaller.
    // We expect roughly 8 * (1 - 0.25) = 6 edges.
    std::cout << "New num nodes: " << new_x.size(0)
              << ", New num edges: " << new_edge_index.size(1) << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    EdgeDrop::EdgeDrop(double drop_rate, double p)
            : drop_rate_(drop_rate), p_(p) {

        if (drop_rate_ < 0.0 || drop_rate_ > 1.0) {
            throw std::invalid_argument("drop_rate must be in [0, 1].");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto EdgeDrop::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability Check ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("EdgeDrop expects 2 tensors: {node_features, edge_index}.");
        }

        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || drop_rate_ == 0.0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        // --- 2. Drop Edges ---
        long num_edges = edge_index.size(1);
        if (num_edges == 0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        // Create a random mask for the edges to keep.
        auto edge_mask = torch::rand({num_edges}, x.options()) > drop_rate_;

        // Select the columns (edges) from edge_index where the mask is true.
        // `torch::where` returns the indices of the true elements.
        // auto keep_indices = std::get<0>(torch::where(edge_mask));
        auto keep_indices = torch::where(edge_mask)[0];

        auto new_edge_index = edge_index.index_select(/*dim=*/1, keep_indices);

        // --- 3. Return the new graph structure ---
        // Node features are unchanged.
        return std::vector<torch::Tensor>{x, new_edge_index};
    }

} // namespace xt::transforms::graph