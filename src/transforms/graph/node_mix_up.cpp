#include <transforms/graph/node_mix_up.h>


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph with easily identifiable features.
    torch::Tensor x = torch::eye(10); // 10 nodes, one-hot features
    torch::Tensor edge_index = torch::tensor({{0, 1}, {1, 0}}, torch::kLong);

    std::cout << "Original node features (first 4 nodes):\n" << x.slice(0, 0, 4) << std::endl;

    // 2. Create the NodeMixUp transform to mix 40% of the nodes.
    xt::transforms::graph::NodeMixUp mixer(0.2, 0.4);

    // 3. Apply the transform.
    auto result_any = mixer.forward({x, edge_index});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = new_graph[0];

    // 4. Verify the output. Some features should be interpolated.
    std::cout << "\nMixed node features (first 4 nodes):\n" << new_x.slice(0, 0, 4) << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    NodeMixUp::NodeMixUp(double alpha, double mixup_ratio, double p)
            : alpha_(alpha), mixup_ratio_(mixup_ratio), p_(p) {

        if (alpha_ <= 0.0) throw std::invalid_argument("alpha must be positive.");
        if (mixup_ratio_ < 0.0 || mixup_ratio_ > 1.0) {
            throw std::invalid_argument("mixup_ratio must be in [0, 1].");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto NodeMixUp::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("NodeMixUp expects 2 tensors: {node_features, edge_index}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || mixup_ratio_ == 0.0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        long num_nodes = x.size(0);
        if (num_nodes < 2) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        // --- 2. Select Nodes to Mix ---
        long num_to_mix = static_cast<long>(num_nodes * mixup_ratio_);
        if (num_to_mix < 2) {
            return std::vector<torch::Tensor>{x, edge_index};
        }
        num_to_mix -= (num_to_mix % 2); // Ensure it's an even number for pairing

        std::vector<long> node_indices(num_nodes);
        std::iota(node_indices.begin(), node_indices.end(), 0);
        std::shuffle(node_indices.begin(), node_indices.end(), random_engine_);

        auto mix_indices_tensor = torch::tensor(
                std::vector<long>(node_indices.begin(), node_indices.begin() + num_to_mix),
                torch::kLong
        ).to(x.device());

        // --- 3. Pair Up Nodes ---
        // Split the indices into two halves for pairing
        auto pair1_indices = mix_indices_tensor.slice(0, 0, num_to_mix / 2);
        auto pair2_indices = mix_indices_tensor.slice(0, num_to_mix / 2);

        // --- 4. Sample Lambda and Mix Features ---
        // Sample one lambda value for all pairs for efficiency
        torch::Tensor alpha_tensor = torch::tensor(alpha_, x.options());
        auto gamma1 = torch::special::gammaln(alpha_tensor);
        auto gamma2 = torch::special::gammaln(alpha_tensor);
        double lambda = (gamma1 / (gamma1 + gamma2)).item<double>();

        auto new_x = x.clone();

        // Get feature vectors for the pairs
        auto x_pair1 = x.index_select(0, pair1_indices);
        auto x_pair2 = x.index_select(0, pair2_indices);

        // Perform the mixup
        auto mixed_features1 = lambda * x_pair1 + (1.0 - lambda) * x_pair2;
        auto mixed_features2 = lambda * x_pair2 + (1.0 - lambda) * x_pair1;

        // Put the new features back into the feature matrix
        new_x.index_put_({pair1_indices}, mixed_features1);
        new_x.index_put_({pair2_indices}, mixed_features2);

        // --- 5. Return the new graph structure ---
        // The edge_index is unchanged.
        return std::vector<torch::Tensor>{new_x, edge_index};
    }

} // namespace xt::transforms::graph