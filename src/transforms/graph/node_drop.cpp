#include "include/transforms/graph/node_drop.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph.
    torch::Tensor x = torch::arange(0, 10, torch::kFloat32).view({-1, 1}); // 10 nodes
    torch::Tensor edge_index = torch::tensor({
        {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9},
        {1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8}
    }, torch::kLong); // A path graph

    std::cout << "Original num nodes: " << x.size(0)
              << ", Original num edges: " << edge_index.size(1) << std::endl;

    // 2. Create the transform to drop 30% of the nodes.
    xt::transforms::graph::NodeDrop node_dropper(0.3);

    // 3. Apply the transform.
    auto result_any = node_dropper.forward({x, edge_index});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = new_graph[0];
    torch::Tensor new_edge_index = new_graph[1];

    // 4. Verify the output. The number of nodes and edges should be smaller.
    // We expect 10 * (1 - 0.3) = 7 nodes.
    std::cout << "New num nodes: " << new_x.size(0)
              << ", New num edges: " << new_edge_index.size(1) << std::endl;
    std::cout << "New node features (should be a subset of original and re-indexed):\n" << new_x << std::endl;
    std::cout << "New edge index (re-indexed):\n" << new_edge_index << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    NodeDrop::NodeDrop(double drop_rate, double p)
            : drop_rate_(drop_rate), p_(p) {

        if (drop_rate_ < 0.0 || drop_rate_ >= 1.0) {
            throw std::invalid_argument("drop_rate must be in [0, 1).");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto NodeDrop::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("NodeDrop expects 2 tensors: {node_features, edge_index}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || drop_rate_ == 0.0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        long num_nodes = x.size(0);
        if (num_nodes == 0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        // --- 2. Select Nodes to Keep ---
        auto keep_mask = torch::rand({num_nodes}, x.options()) > drop_rate_;
        // auto keep_indices = std::get<0>(torch::where(keep_mask));
        auto keep_indices = torch::where(keep_mask)[0];


        if (keep_indices.numel() == 0) {
            // Avoid creating an empty graph, return a graph with one node.
            auto new_x = x.slice(0, 0, 1).clone();
            auto new_edge_index = torch::empty({2, 0}, edge_index.options());
            return std::vector<torch::Tensor>{new_x, new_edge_index};
        }

        // --- 3. Filter Node Features ---
        auto new_x = x.index_select(0, keep_indices);

        // --- 4. Create Mapping from Old to New Indices ---
        auto mapping = torch::full({num_nodes}, -1, torch::kLong).to(x.device());
        mapping.index_put_({keep_indices}, torch::arange(0, keep_indices.numel(), torch::kLong).to(x.device()));

        // --- 5. Filter Edges ---
        // An edge is kept if both its source and destination nodes are kept.
        auto row = edge_index[0];
        auto col = edge_index[1];
        auto row_mask = keep_mask.index({row});
        auto col_mask = keep_mask.index({col});
        auto edge_keep_mask = row_mask & col_mask;
        // auto keep_indices = torch::where(edge_mask)[0];

        auto kept_edges = edge_index.index_select(1, torch::where(edge_keep_mask)[0]);

        // --- 6. Re-index Kept Edges ---
        if (kept_edges.size(1) > 0) {
            auto new_row = mapping.index({kept_edges[0]});
            auto new_col = mapping.index({kept_edges[1]});
            auto new_edge_index = torch::stack({new_row, new_col}, 0);
            return std::vector<torch::Tensor>{new_x, new_edge_index};
        } else {
            // No edges left after dropping nodes
            auto new_edge_index = torch::empty({2, 0}, edge_index.options());
            return std::vector<torch::Tensor>{new_x, new_edge_index};
        }
    }

} // namespace xt::transforms::graph