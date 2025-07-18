#include <transforms/graph/graph_mix_up.h>



/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create two dummy graphs of different sizes.
    torch::Tensor x1 = torch::ones({5, 8}); // 5 nodes
    torch::Tensor edge_index1 = torch::tensor({{0, 1}, {1, 0}}, torch::kLong);

    torch::Tensor x2 = torch::ones({8, 8}) * 2.0; // 8 nodes
    torch::Tensor edge_index2 = torch::tensor({{0, 1, 2, 3}, {1, 2, 3, 0}}, torch::kLong);

    std::cout << "Graph 1: " << x1.size(0) << " nodes, " << edge_index1.size(1) << " edges." << std::endl;
    std::cout << "Graph 2: " << x2.size(0) << " nodes, " << edge_index2.size(1) << " edges." << std::endl;

    // 2. Create the GraphMixUp transform.
    xt::transforms::graph::GraphMixUp mixer(0.2);

    // 3. Apply the transform.
    auto result_any = mixer.forward({x1, edge_index1, x2, edge_index2});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = new_graph[0];
    torch::Tensor new_edge_index = new_graph[1];

    // 4. Verify the output.
    // The new graph should have max(5, 8) = 8 nodes.
    // The new graph should have 2 + 4 = 6 edges.
    // Node features should be interpolated.
    std::cout << "\nMixed Graph: " << new_x.size(0) << " nodes, "
              << new_edge_index.size(1) << " edges." << std::endl;
    std::cout << "First row of mixed features (should be interpolated):\n" << new_x[0] << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    GraphMixUp::GraphMixUp(double alpha, double p)
            : alpha_(alpha), p_(p) {

        if (alpha_ <= 0.0) {
            throw std::invalid_argument("alpha must be positive.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto GraphMixUp::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 4) {
            throw std::invalid_argument("GraphMixUp expects 4 tensors: {x1, edge_index1, x2, edge_index2}.");
        }
        torch::Tensor x1 = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index1 = std::any_cast<torch::Tensor>(any_vec[1]);
        torch::Tensor x2 = std::any_cast<torch::Tensor>(any_vec[2]);
        torch::Tensor edge_index2 = std::any_cast<torch::Tensor>(any_vec[3]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_) {
            // If not applying, just return the first graph as-is.
            return std::vector<torch::Tensor>{x1, edge_index1};
        }

        // --- 2. Sample Lambda from Beta Distribution ---
        // Since LibTorch doesn't have a direct Beta distribution sampler, we can
        // construct one from two Gamma distributions.
        // lambda ~ Beta(alpha, alpha)
        torch::Tensor alpha_tensor = torch::tensor(alpha_, x1.options());
        auto gamma1 = torch::special::gammaln(alpha_tensor); // Using gamma as a proxy
        auto gamma2 = torch::special::gammaln(alpha_tensor);
        double lambda = (gamma1 / (gamma1 + gamma2)).item<double>();

        // --- 3. Align Node Features ---
        long n1 = x1.size(0);
        long n2 = x2.size(0);
        long num_features = x1.size(1);

        if (x1.size(1) != x2.size(1)) {
            throw std::invalid_argument("Node features of both graphs must have the same dimension.");
        }

        // Pad the smaller feature matrix with zeros to match the larger one.
        if (n1 < n2) {
            auto padding = torch::zeros({n2 - n1, num_features}, x1.options());
            x1 = torch::cat({x1, padding}, 0);
        } else if (n2 < n1) {
            auto padding = torch::zeros({n1 - n2, num_features}, x2.options());
            x2 = torch::cat({x2, padding}, 0);
        }

        // --- 4. Interpolate Node Features ---
        torch::Tensor new_x = lambda * x1 + (1.0 - lambda) * x2;

        // --- 5. Combine Edge Indices ---
        // The new edge set is the union of the two edge sets.
        torch::Tensor new_edge_index = torch::cat({edge_index1, edge_index2}, 1);

        // Optional: Remove duplicate edges that might result from the union.
        // This is the same technique used in GraphCoarsening.
        long num_new_nodes = new_x.size(0);
        auto edge_keys = new_edge_index[0] * num_new_nodes + new_edge_index[1];
        auto unique_keys_and_indices = torch::_unique(edge_keys);
        auto unique_indices = std::get<1>(unique_keys_and_indices);
        new_edge_index = new_edge_index.index_select(1, unique_indices);

        return std::vector<torch::Tensor>{new_x, new_edge_index};
    }

} // namespace xt::transforms::graph