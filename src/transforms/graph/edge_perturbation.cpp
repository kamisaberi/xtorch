#include "include/transforms/graph/edge_perturbation.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph.
    torch::Tensor x = torch::randn({10, 8});
    torch::Tensor edge_index = torch::tensor({
        {0, 1, 1, 2, 2, 3, 3, 4, 4, 5},
        {1, 0, 2, 1, 3, 2, 4, 3, 5, 4}
    }, torch::kLong); // 10 edges

    std::cout << "Original num edges: " << edge_index.size(1) << std::endl;

    // 2. Create the transform to drop 20% and add 20% of the original number of edges.
    xt::transforms::graph::EdgePerturbation perturber(0.2, 0.2);

    // 3. Apply the transform.
    auto result_any = perturber.forward({x, edge_index});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_edge_index = new_graph[1];

    // 4. Verify the output.
    // We expect 10 * (1 - 0.2) + 10 * 0.2 = 8 + 2 = 10 edges.
    // The number of edges should be roughly the same, but the connections will be different.
    std::cout << "New num edges: " << new_edge_index.size(1) << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    EdgePerturbation::EdgePerturbation(double add_ratio, double drop_ratio, double p)
            : add_ratio_(add_ratio), drop_ratio_(drop_ratio), p_(p) {

        if (add_ratio_ < 0.0 || drop_ratio_ < 0.0 || drop_ratio_ > 1.0) {
            throw std::invalid_argument("Invalid add_ratio or drop_ratio.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto EdgePerturbation::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability Check ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("EdgePerturbation expects 2 tensors: {node_features, edge_index}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || (add_ratio_ == 0.0 && drop_ratio_ == 0.0)) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        long num_nodes = x.size(0);
        long num_edges = edge_index.size(1);

        // --- 2. Drop Edges ---
        auto keep_mask = torch::rand({num_edges}, x.options()) > drop_ratio_;
        auto keep_indices = std::get<0>(torch::where(keep_mask));
        auto remaining_edge_index = edge_index.index_select(/*dim=*/1, keep_indices);

        // --- 3. Add Edges ---
        long num_edges_to_add = static_cast<long>(num_edges * add_ratio_);
        if (num_edges_to_add == 0 || num_nodes < 2) {
            return std::vector<torch::Tensor>{x, remaining_edge_index};
        }

        // --- Create a Hash Set of ALL original edges for efficient lookup ---
        std::unordered_set<long long> existing_edges;
        auto edge_index_cpu = edge_index.to(torch::kCPU).contiguous();
        auto accessor = edge_index_cpu.accessor<long, 2>();
        for (long i = 0; i < num_edges; ++i) {
            existing_edges.insert(
                    accessor[0][i] * num_nodes + accessor[1][i]
            );
        }

        // --- Generate New Edges ---
        std::vector<torch::Tensor> new_edges_list;
        new_edges_list.reserve(num_edges_to_add);
        std::uniform_int_distribution<long> node_dist(0, num_nodes - 1);

        int max_tries = num_edges_to_add * 20;
        int tries = 0;
        while (new_edges_list.size() < num_edges_to_add && tries < max_tries) {
            long u = node_dist(random_engine_);
            long v = node_dist(random_engine_);
            long long key = u * num_nodes + v;

            if (u == v || existing_edges.count(key)) {
                tries++;
                continue;
            }

            new_edges_list.push_back(torch::tensor({u, v}, torch::kLong));
            existing_edges.insert(key); // Add to set to prevent re-adding the same new edge
            tries = 0;
        }

        // --- 4. Combine and Return ---
        if (new_edges_list.empty()) {
            return std::vector<torch::Tensor>{x, remaining_edge_index};
        }

        torch::Tensor new_edges_tensor = torch::stack(new_edges_list, 1).to(edge_index.device());
        torch::Tensor final_edge_index = torch::cat({remaining_edge_index, new_edges_tensor}, 1);

        return std::vector<torch::Tensor>{x, final_edge_index};
    }

} // namespace xt::transforms::graph