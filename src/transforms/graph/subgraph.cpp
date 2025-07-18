#include <transforms/graph/subgraph.h>


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph.
    torch::Tensor x = torch::arange(0, 5, torch::kFloat32).view({-1, 1}); // 5 nodes
    torch::Tensor edge_index = torch::tensor({
        {0, 1, 1, 2, 2, 3, 3, 0},
        {1, 0, 2, 1, 3, 2, 0, 3}
    }, torch::kLong);

    std::cout << "Original Graph:" << std::endl;
    std::cout << "  - Nodes:\n" << x << std::endl;
    std::cout << "  - Edges:\n" << edge_index << std::endl;


    // 2. Define the nodes we want to keep in our subgraph.
    torch::Tensor nodes_to_keep = torch::tensor({0, 1, 3}, torch::kLong);
    std::cout << "\nExtracting subgraph with nodes: " << nodes_to_keep << std::endl;

    // 3. Create and apply the transform.
    xt::transforms::graph::Subgraph subgraph_extractor;
    auto result_any = subgraph_extractor.forward({x, edge_index, nodes_to_keep});
    auto subgraph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = subgraph[0];
    torch::Tensor new_edge_index = subgraph[1];

    // 4. Verify the output.
    // New nodes should be [0, 1, 3]. Edges should be (0,1), (1,0), (3,0), (0,3).
    // Re-indexing: 0->0, 1->1, 3->2. New edges: (0,1), (1,0), (2,0), (0,2).
    std::cout << "\nResulting Subgraph:" << std::endl;
    std::cout << "  - Nodes:\n" << new_x << std::endl;
    std::cout << "  - Edges (re-indexed):\n" << new_edge_index << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph
{
    Subgraph::Subgraph()
    {
    } // No parameters needed for this transform

    auto Subgraph::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        // --- 1. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 3)
        {
            throw std::invalid_argument("Subgraph expects 3 tensors: {node_features, edge_index, nodes_to_keep}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);
        torch::Tensor nodes_to_keep = std::any_cast<torch::Tensor>(any_vec[2]);

        if (!x.defined() || !edge_index.defined() || !nodes_to_keep.defined())
        {
            throw std::invalid_argument("All input tensors must be defined.");
        }
        if (nodes_to_keep.dim() != 1 || nodes_to_keep.dtype() != torch::kLong)
        {
            throw std::invalid_argument("nodes_to_keep must be a 1D tensor of type long.");
        }

        long num_nodes = x.size(0);

        // --- 2. Filter Node Features ---
        // Select the rows from the feature matrix corresponding to the desired nodes.
        auto new_x = x.index_select(0, nodes_to_keep);

        // --- 3. Create a Mask for Kept Nodes ---
        // This mask allows for efficient filtering of edges.
        auto node_mask = torch::zeros({num_nodes}, torch::kBool).to(x.device());
        node_mask.index_fill_(0, nodes_to_keep, true);

        // --- 4. Filter Edges ---
        // An edge is kept if and only if both its source and destination nodes
        // are in the `nodes_to_keep` set.
        auto row = edge_index[0];
        auto col = edge_index[1];
        auto edge_keep_mask = node_mask.index({row}) & node_mask.index({col});

        auto kept_edges = edge_index.index_select(1, torch::where(edge_keep_mask)[0]);

        // --- 5. Re-index Kept Edges ---
        if (kept_edges.size(1) > 0)
        {
            // Create a mapping from old node indices to new contiguous indices (0, 1, 2, ...).
            auto mapping = torch::full({num_nodes}, -1, torch::kLong).to(x.device());
            mapping.index_put_({nodes_to_keep}, torch::arange(0, nodes_to_keep.numel(), torch::kLong).to(x.device()));

            // Apply the mapping to the filtered edges.
            auto new_row = mapping.index({kept_edges[0]});
            auto new_col = mapping.index({kept_edges[1]});
            auto new_edge_index = torch::stack({new_row, new_col}, 0);
            return std::vector<torch::Tensor>{new_x, new_edge_index};
        }
        else
        {
            // If no edges remain, return an empty edge index tensor.
            auto new_edge_index = torch::empty({2, 0}, edge_index.options());
            return std::vector<torch::Tensor>{new_x, new_edge_index};
        }
    }
} // namespace xt::transforms::graph
