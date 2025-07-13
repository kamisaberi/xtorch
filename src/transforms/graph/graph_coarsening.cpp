#include "include/transforms/graph/graph_coarsening.h"



/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph (e.g., a simple path graph).
    torch::Tensor x = torch::randn({10, 8}); // 10 nodes
    torch::Tensor edge_index = torch::tensor({
        {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9},
        {1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8}
    }, torch::kLong); // 9 bidirectional edges

    std::cout << "Original num nodes: " << x.size(0)
              << ", Original num edges: " << edge_index.size(1) << std::endl;

    // 2. Create the transform to coarsen the graph to ~50% of its size.
    xt::transforms::graph::GraphCoarsening coarsey(0.5);

    // 3. Apply the transform.
    auto result_any = coarsey.forward({x, edge_index});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = new_graph[0];
    torch::Tensor new_edge_index = new_graph[1];

    // 4. Verify the output. The number of nodes should be smaller.
    std::cout << "New num nodes: " << new_x.size(0)
              << ", New num edges: " << new_edge_index.size(1) << std::endl;
    // Expected new nodes: ~5. Edges will be reconnected.

    return 0;
}
*/

namespace xt::transforms::graph {

    GraphCoarsening::GraphCoarsening(double coarsening_ratio, double p)
            : coarsening_ratio_(coarsening_ratio), p_(p) {

        if (coarsening_ratio_ <= 0.0 || coarsening_ratio_ > 1.0) {
            throw std::invalid_argument("coarsening_ratio must be in (0, 1].");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto GraphCoarsening::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("GraphCoarsening expects 2 tensors: {node_features, edge_index}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || coarsening_ratio_ == 1.0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        long num_nodes = x.size(0);
        if (num_nodes == 0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        // --- 2. Greedy Node Matching ---
        // This simple algorithm iterates through nodes in random order and pairs
        // each un-matched node with a random un-matched neighbor.
        std::vector<long> perm(num_nodes);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), random_engine_);

        // `cluster` will map each old node to its new supernode index.
        auto cluster = torch::full({num_nodes}, -1, torch::kLong);
        long new_node_idx = 0;

        // Convert edge_index to an adjacency list for easy neighbor lookup.
        std::vector<std::vector<long>> adj(num_nodes);
        auto edge_index_cpu = edge_index.to(torch::kCPU).contiguous();
        for(long i=0; i<edge_index.size(1); ++i) {
            adj[edge_index_cpu[0][i].item<long>()].push_back(edge_index_cpu[1][i].item<long>());
        }

        for (long i : perm) {
            if (cluster[i].item<long>() == -1) { // If node i is not yet clustered
                // Find a random un-clustered neighbor
                long best_neighbor = -1;
                std::shuffle(adj[i].begin(), adj[i].end(), random_engine_);
                for (long neighbor : adj[i]) {
                    if (cluster[neighbor].item<long>() == -1) {
                        best_neighbor = neighbor;
                        break;
                    }
                }

                if (best_neighbor != -1 && new_node_idx < num_nodes * coarsening_ratio_) {
                    cluster[i] = new_node_idx;
                    cluster[best_neighbor] = new_node_idx;
                } else {
                    // Node has no un-matched neighbors or we've hit our ratio, it becomes its own cluster.
                    cluster[i] = new_node_idx;
                }
                new_node_idx++;
            }
        }

        long num_new_nodes = new_node_idx;
        cluster = cluster.to(x.device());

        // --- 3. Pool Node Features ---
        // We use scatter_add to sum the features of nodes in the same cluster.
        auto new_x = torch::zeros({num_new_nodes, x.size(1)}, x.options());
        new_x.scatter_add_(0, cluster.unsqueeze(1).expand_as(x), x);

        // Normalize by cluster size to get the mean feature.
        auto cluster_sizes = torch::bincount(cluster, {}, num_new_nodes).to(x.dtype());
        new_x = new_x / cluster_sizes.unsqueeze(1).clamp_min(1.0);

        // --- 4. Reconstruct Edge Index ---
        // Map old edges to new supernode edges.
        auto row = edge_index[0];
        auto col = edge_index[1];
        row = cluster.index({row});
        col = cluster.index({col});

        // Create the new edge index, removing self-loops.
        auto mask = row != col;
        auto new_edge_index = torch::stack({row.index({mask}), col.index({mask})}, 0);

        // Remove duplicate edges. This is a bit tricky but effective.
        // It works by converting edge pairs to unique numbers, finding unique ones,
        // and then converting back.
        auto edge_keys = new_edge_index[0] * num_new_nodes + new_edge_index[1];
        auto unique_keys_and_indices = torch::_unique(edge_keys);
        auto unique_indices = std::get<1>(unique_keys_and_indices);
        new_edge_index = new_edge_index.index_select(1, unique_indices);

        return std::vector<torch::Tensor>{new_x, new_edge_index};
    }

} // namespace xt::transforms::graph