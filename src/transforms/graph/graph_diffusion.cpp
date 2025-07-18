#include <transforms/graph/graph_diffusion.h>
#include <unordered_set>

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
        torch::Tensor current_edge_index = edge_index;
        if (add_self_loops_) {
            auto self_loops = torch::arange(0, N, torch::kLong).to(device);
            self_loops = self_loops.unsqueeze(0).repeat({2, 1});
            current_edge_index = torch::cat({edge_index, self_loops}, 1);
        }

        auto edge_values = torch::ones({current_edge_index.size(1)}, options);
        auto A = torch::zeros({N, N}, options);
        A.index_put_({current_edge_index[0], current_edge_index[1]}, edge_values);

        auto D_inv_sqrt = torch::pow(A.sum(1), -0.5);
        D_inv_sqrt.masked_fill_(torch::isinf(D_inv_sqrt), 0);
        auto D_inv_sqrt_diag = torch::diag(D_inv_sqrt);

        // --- 3. Compute Normalized Laplacian ---
        auto I = torch::eye(N, options);
        auto L = I - torch::mm(torch::mm(D_inv_sqrt_diag, A), D_inv_sqrt_diag);

        // --- 4. Approximate Diffusion Operator via Taylor Series ---
        // P = exp(-beta * L) approx_equal_to sum_{i=0 to k-1} (-beta * L)^i / i!
        // We calculate each term T_i = (-beta*L)^i / i! from the previous term T_{i-1}.
        // T_i = T_{i-1} * (-beta*L) / i
        torch::Tensor P = I.clone(); // P starts with the i=0 term (I)
        torch::Tensor T_i = I.clone(); // T_0 = I
        auto M = -beta_ * L;

        for (int i = 1; i < k_; ++i) {
            T_i = torch::mm(T_i, M) / static_cast<double>(i);
            P += T_i;
        }

        // --- 5. Apply Diffusion to Features ---
        auto new_x = torch::mm(P, x);

        // --- 6. Optionally Add New Edges ---
        auto new_edge_index = edge_index;
        if (add_new_edges_threshold_ > 0.0 && add_new_edges_threshold_ < 1.0) {
            auto diffused_adj = P.clone();
            diffused_adj.fill_diagonal_(0); // Remove self-loops from consideration

            // torch::where on a 2D tensor returns a std::vector of two tensors: {row_indices, col_indices}
            auto new_indices_vec = torch::where(diffused_adj > add_new_edges_threshold_);

            if (!new_indices_vec.empty() && new_indices_vec[0].numel() > 0) {
                auto u_indices = new_indices_vec[0];
                auto v_indices = new_indices_vec[1];

                // Filter out edges that already exist using a hash set.
                std::unordered_set<long long> existing_edges;
                auto edge_index_cpu = edge_index.to(torch::kCPU).contiguous();
                auto u_existing_ptr = edge_index_cpu[0].data_ptr<long>();
                auto v_existing_ptr = edge_index_cpu[1].data_ptr<long>();
                for(long i = 0; i < edge_index.size(1); ++i) {
                    existing_edges.insert(u_existing_ptr[i] * N + v_existing_ptr[i]);
                }

                std::vector<long> u_to_add;
                std::vector<long> v_to_add;
                auto u_new_cpu = u_indices.to(torch::kCPU).contiguous();
                auto v_new_cpu = v_indices.to(torch::kCPU).contiguous();
                auto u_new_ptr = u_new_cpu.data_ptr<long>();
                auto v_new_ptr = v_new_cpu.data_ptr<long>();

                for(long i = 0; i < u_indices.size(0); ++i) {
                    long u = u_new_ptr[i];
                    long v = v_new_ptr[i];
                    if (existing_edges.find(u * N + v) == existing_edges.end()) {
                        u_to_add.push_back(u);
                        v_to_add.push_back(v);
                    }
                }

                if (!u_to_add.empty()) {
                    auto u_tensor = torch::tensor(u_to_add, torch::kLong);
                    auto v_tensor = torch::tensor(v_to_add, torch::kLong);
                    auto edges_to_add_tensor = torch::stack({u_tensor, v_tensor}).to(device);
                    new_edge_index = torch::cat({edge_index, edges_to_add_tensor}, 1);
                }
            }
        }

        return std::vector<torch::Tensor>{new_x, new_edge_index};
    }

} // namespace xt::transforms::graph