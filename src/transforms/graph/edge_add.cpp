#include <transforms/graph/edge_add.h>


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph.
    // Node features: 10 nodes, 16 features per node.
    torch::Tensor x = torch::randn({10, 16});
    // Edge index: 8 edges.
    torch::Tensor edge_index = torch::tensor({
        {0, 1, 1, 2, 2, 3, 3, 4},
        {1, 0, 2, 1, 3, 2, 4, 3}
    }, torch::kLong);

    std::cout << "Original num nodes: " << x.size(0)
              << ", Original num edges: " << edge_index.size(1) << std::endl;

    // 2. Create the transform to add 50% new edges.
    xt::transforms::graph::EdgeAdd edge_adder(0.5);

    // 3. Apply the transform.
    auto result_any = edge_adder.forward({x, edge_index});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = new_graph[0];
    torch::Tensor new_edge_index = new_graph[1];

    // 4. Verify the output.
    // We expect 8 * 0.5 = 4 new edges, for a total of 12.
    std::cout << "New num nodes: " << new_x.size(0)
              << ", New num edges: " << new_edge_index.size(1) << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    EdgeAdd::EdgeAdd(double add_ratio, double p)
            : add_ratio_(add_ratio), p_(p) {

        if (add_ratio_ < 0.0) {
            throw std::invalid_argument("add_ratio must be non-negative.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto EdgeAdd::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability Check ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("EdgeAdd expects 2 tensors: {node_features, edge_index}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || add_ratio_ == 0.0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        // --- 2. Setup ---
        long num_nodes = x.size(0);
        long num_edges = edge_index.size(1);
        long num_edges_to_add = static_cast<long>(num_edges * add_ratio_);

        if (num_edges_to_add == 0 || num_nodes < 2) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        // --- 3. Create a Hash Set of Existing Edges for Efficient Lookup ---
        // We encode an edge (u, v) as a single integer: u * num_nodes + v.
        std::unordered_set<long long> existing_edges;
        auto edge_index_cpu = edge_index.to(torch::kCPU).contiguous();
        auto accessor = edge_index_cpu.accessor<long, 2>();
        for (long i = 0; i < num_edges; ++i) {
            existing_edges.insert(
                    accessor[0][i] * num_nodes + accessor[1][i]
            );
        }

        // --- 4. Generate New Edges ---
        std::vector<torch::Tensor> new_edges_list;
        new_edges_list.reserve(num_edges_to_add);
        std::uniform_int_distribution<long> node_dist(0, num_nodes - 1);

        // To avoid an infinite loop on a dense graph, we add a try limit.
        int max_tries = num_edges_to_add * 20;
        int tries = 0;

        while (new_edges_list.size() < num_edges_to_add && tries < max_tries) {
            long u = node_dist(random_engine_);
            long v = node_dist(random_engine_);
            long long key = u * num_nodes + v;

            if (u == v || existing_edges.count(key)) {
                tries++;
                continue; // Skip self-loops and existing edges
            }

            // Add the new edge
            new_edges_list.push_back(torch::tensor({u, v}, torch::kLong));
            existing_edges.insert(key);
            tries = 0; // Reset tries after a success
        }

        // --- 5. Combine and Return ---
        if (new_edges_list.empty()) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        torch::Tensor new_edges_tensor = torch::stack(new_edges_list, 1).to(edge_index.device());
        torch::Tensor new_edge_index = torch::cat({edge_index, new_edges_tensor}, 1);

        return std::vector<torch::Tensor>{x, new_edge_index};
    }

} // namespace xt::transforms::graph