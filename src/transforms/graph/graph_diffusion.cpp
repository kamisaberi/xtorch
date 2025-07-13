#include "include/transforms/graph/graph_diffusion.h"


// It's good practice to include utilities for sparse tensors if available.
// Libtorch has some sparse support, but it can be limited. We'll use dense here
// for simplicity, but a production version would use sparse matrices.

/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph with some structure.
    torch::Tensor x = torch::eye(5); // One-hot features for 5 nodes
    torch::Tensor edge_index = torch::tensor({{0, 1, 1, 2, 3, 4}, {1, 0, 2, 1, 4, 3}}, torch::kLong);

    std::cout << "Original Node Features:\n" << x << std::endl;
    std::cout << "Original Edge Index:\n" << edge_index << std::endl;

    // 2. Create the transform to diffuse the graph.
    // Use a threshold to also add new, high-probability edges.
    xt::transforms::graph::GraphDiffusion diffuser(1.0, true, 8, 0.2);

    // 3. Apply the transform.
    auto result_any = diffuser.forward({x, edge_index});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = new_graph[0];
    torch::Tensor new_edge_index = new_graph[1];

    // 4. Verify the output. Features should be smoothed, and new edges might be added.
    std::cout << "\nDiffused Node Features (smoothed):\n" << new_x << std::endl;
    std::cout << "New Edge Index (may include new edges like 0-2):\n" << new_edge_index << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    GraphDiffusion::GraphDiffusion(double beta, bool add_self_loops, int k, double add_new_edges_threshold)
            : beta_(beta), add_self_loops_(add_self_loops), k_(k), add_new_edges_threshold_(add_new_edges_threshold) {

        if (beta_ < 0) throw std::invalid_argument("beta must be non-negative.");
        if (k_ < 1) throw std::invalid_argument("k must be at least 1.");
        if (add_new_edges_threshold_ < 0.0 || add_new_edges_threshold_ > 1.0) {
            throw std::invalid_argument("add_new_edges_threshold must be in [0, 1].");
        }
    }

    auto GraphDiffusion::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("GraphDiffusion expects 2 tensors: {node_features, edge_index}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        long N = x.size(0);
        if (N == 0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }
        auto device = x.device();
        auto options = x.options();

        // --- 2. Build Adjacency Matrix and Degree Vector ---
        auto current_edge_index = edge_index;
        if (add_self_loops_) {
            auto self_loops = torch::arange(0, N, torch::kLong).to(device);
            self_loops = self_loops.unsqueeze(0).repeat({2, 1});
            current_edge_index = torch::cat({edge_index, self_loops}, 1);
        }

        auto edge_values = torch::ones({current_edge_index.size(1)}, options);
        // Note: For large graphs, this should be a sparse matrix.
        auto A = torch::zeros({N, N}, options);
        A.index_put_({current_edge_index[0], current_edge_index[1]}, edge_values);

        auto D_inv_sqrt = torch::pow(A.sum(1), -0.5);
        D_inv_sqrt.masked_fill_(torch::isinf(D_inv_sqrt), 0);
        auto D_inv_sqrt_diag = torch::diag(D_inv_sqrt);

        // --- 3. Compute Normalized Laplacian ---
        // L = I - D^(-1/2) * A * D^(-1/2)
        auto I = torch::eye(N, options);
        auto L = I - torch::mm(torch::mm(D_inv_sqrt_diag, A), D_inv_sqrt_diag);

        // --- 4. Approximate Diffusion Operator via Taylor Series ---
        // P = exp(-beta * L) approx_equal_to sum_{i=0 to k} (-beta * L)^i / i!
        torch::Tensor P = torch::zeros_like(L);
        torch::Tensor L_pow_i = torch::eye(N, options); // L^0
        double factorial_i = 1.0;

        for (int i = 0; i < k_; ++i) {
            P += L_pow_i / factorial_i;
            L_pow_i = torch::mm(L_pow_i, -beta_ * L);
            if (i > 0) factorial_i *= i;
        }

        // --- 5. Apply Diffusion to Features ---
        auto new_x = torch::mm(P, x);

        // --- 6. Optionally Add New Edges ---
        auto new_edge_index = edge_index;
        if (add_new_edges_threshold_ > 0.0) {
            // Find entries in P above the threshold
            auto diffused_adj = P.clone();
            diffused_adj.fill_diagonal_(0); // Remove self-loops from consideration
            auto new_possible_edges = std::get<0>(torch::where(diffused_adj > add_new_edges_threshold_));

            // To be safe, filter out edges that already exist
            std::unordered_set<long long> existing_edges;
            auto edge_index_cpu = edge_index.to(torch::kCPU).contiguous();
            for(long i=0; i<edge_index.size(1); ++i) {
                existing_edges.insert(
                        edge_index_cpu[0][i].item<long>() * N + edge_index_cpu[1][i].item<long>()
                );
            }

            std::vector<torch::Tensor> edges_to_add_list;
            for(long i=0; i<new_possible_edges.size(0); i += 2) {
                long u = new_possible_edges[i].item<long>();
                long v = new_possible_edges[i+1].item<long>();
                if (existing_edges.find(u * N + v) == existing_edges.end()) {
                    edges_to_add_list.push_back(torch::tensor({u, v}, torch::kLong));
                }
            }

            if (!edges_to_add_list.empty()) {
                auto edges_to_add_tensor = torch::stack(edges_to_add_list, 1).to(device);
                new_edge_index = torch::cat({edge_index, edges_to_add_tensor}, 1);
            }
        }

        return std::vector<torch::Tensor>{new_x, new_edge_index};
    }

} // namespace xt::transforms::graph