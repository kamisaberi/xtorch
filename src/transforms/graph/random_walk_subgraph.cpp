#include "include/transforms/graph/random_walk_subgraph.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a larger dummy graph (e.g., a grid).
    int grid_size = 5;
    int num_nodes = grid_size * grid_size;
    torch::Tensor x = torch::randn({num_nodes, 8});
    std::vector<torch::Tensor> edge_list;
    for(int i=0; i<grid_size; ++i) {
        for(int j=0; j<grid_size; ++j) {
            int u = i * grid_size + j;
            if (i + 1 < grid_size) {
                int v = (i + 1) * grid_size + j;
                edge_list.push_back(torch::tensor({u, v}, torch::kLong));
                edge_list.push_back(torch::tensor({v, u}, torch::kLong));
            }
            if (j + 1 < grid_size) {
                int v = i * grid_size + (j + 1);
                edge_list.push_back(torch::tensor({u, v}, torch::kLong));
                edge_list.push_back(torch::tensor({v, u}, torch::kLong));
            }
        }
    }
    torch::Tensor edge_index = torch::stack(edge_list, 1);

    std::cout << "Original Graph: " << num_nodes << " nodes, "
              << edge_index.size(1) << " edges." << std::endl;

    // 2. Create the sampler to extract a subgraph starting from 5 nodes with walk length 10.
    xt::transforms::graph::RandomWalkSubgraph sampler(5, 10);

    // 3. Apply the transform.
    auto result_any = sampler.forward({x, edge_index});
    auto subgraph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = subgraph[0];
    torch::Tensor new_edge_index = subgraph[1];

    // 4. Verify the output. The subgraph will be smaller than the original.
    std::cout << "Subgraph: " << new_x.size(0) << " nodes, "
              << new_edge_index.size(1) << " edges." << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    RandomWalkSubgraph::RandomWalkSubgraph(int num_start_nodes, int walk_length)
            : num_start_nodes_(num_start_nodes), walk_length_(walk_length) {

        if (num_start_nodes_ <= 0) throw std::invalid_argument("num_start_nodes must be positive.");
        if (walk_length_ <= 0) throw std::invalid_argument("walk_length must be positive.");

        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto RandomWalkSubgraph::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("RandomWalkSubgraph expects 2 tensors: {node_features, edge_index}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        long num_nodes = x.size(0);
        if (num_nodes == 0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        // --- 2. Build Adjacency List for Efficient Traversal ---
        std::vector<std::vector<long>> adj(num_nodes);
        auto edge_index_cpu = edge_index.to(torch::kCPU).contiguous();
        for(long i=0; i<edge_index.size(1); ++i) {
            adj[edge_index_cpu[0][i].item<long>()].push_back(edge_index_cpu[1][i].item<long>());
        }

        // --- 3. Perform Random Walks ---
        // Select starting nodes
        std::vector<long> all_nodes_vec(num_nodes);
        std::iota(all_nodes_vec.begin(), all_nodes_vec.end(), 0);
        std::shuffle(all_nodes_vec.begin(), all_nodes_vec.end(), random_engine_);

        std::unordered_set<long> visited_nodes;
        for (int i = 0; i < std::min((long)num_start_nodes_, num_nodes); ++i) {
            long current_node = all_nodes_vec[i];
            visited_nodes.insert(current_node);

            for (int step = 0; step < walk_length_; ++step) {
                if (adj[current_node].empty()) break; // Dead end

                // Choose a random neighbor
                std::uniform_int_distribution<size_t> dist(0, adj[current_node].size() - 1);
                current_node = adj[current_node][dist(random_engine_)];
                visited_nodes.insert(current_node);
            }
        }

        // Convert the set of visited nodes to a sorted vector to create a mapping.
        std::vector<long> subgraph_nodes_vec(visited_nodes.begin(), visited_nodes.end());
        std::sort(subgraph_nodes_vec.begin(), subgraph_nodes_vec.end());
        auto subgraph_nodes_tensor = torch::tensor(subgraph_nodes_vec, torch::kLong).to(x.device());

        // --- 4. Extract Subgraph Node Features ---
        auto new_x = x.index_select(0, subgraph_nodes_tensor);

        // --- 5. Extract and Re-index Subgraph Edges ---
        // Create a mapping from old node indices to new subgraph indices.
        auto mapping = torch::full({num_nodes}, -1, torch::kLong).to(x.device());
        mapping.index_put_({subgraph_nodes_tensor}, torch::arange(0, subgraph_nodes_tensor.numel(), torch::kLong).to(x.device()));

        // Filter edges where both source and destination are in the subgraph.
        auto row = edge_index[0];
        auto col = edge_index[1];
        auto row_in_subgraph = torch::from_blob(
                visited_nodes.count(row.data_ptr<long>()) ? (bool*)1 : (bool*)0,
                row.sizes(),
                torch::kBool
        ); // This is a bit of a hack, better to iterate.
        // A safer, more explicit way:
        auto row_mask = torch::zeros({num_nodes}, torch::kBool).to(x.device());
        row_mask.index_fill_(0, subgraph_nodes_tensor, true);
        auto col_mask = row_mask; // Re-use for column
        auto edge_keep_mask = row_mask.index({row}) & col_mask.index({col});

        auto kept_edges = edge_index.index_select(1, std::get<0>(torch::where(edge_keep_mask)));

        // Re-index the kept edges.
        if (kept_edges.size(1) > 0) {
            auto new_row = mapping.index({kept_edges[0]});
            auto new_col = mapping.index({kept_edges[1]});
            auto new_edge_index = torch::stack({new_row, new_col}, 0);
            return std::vector<torch::Tensor>{new_x, new_edge_index};
        } else {
            auto new_edge_index = torch::empty({2, 0}, edge_index.options());
            return std::vector<torch::Tensor>{new_x, new_edge_index};
        }
    }

} // namespace xt::transforms::graph