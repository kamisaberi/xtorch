#include "include/transforms/graph/feature_dropout.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph.
    torch::Tensor x = torch::ones({5, 10}); // 5 nodes, 10 features
    torch::Tensor edge_index = torch::tensor({{0, 1}, {1, 0}}, torch::kLong);

    std::cout << "Sum of features before dropout: " << x.sum().item<float>() << std::endl;

    // 2. Create the transform to drop 30% of the features.
    xt::transforms::graph::FeatureDropout feature_dropper(0.3);

    // 3. Apply the transform.
    auto result_any = feature_dropper.forward({x, edge_index});
    auto new_graph = std::any_cast<std::vector<torch::Tensor>>(result_any);
    torch::Tensor new_x = new_graph[0];

    // 4. Verify the output. The expected sum should be the same as before,
    // but some elements will be zero and others will be scaled up.
    std::cout << "Sum of features after dropout: " << new_x.sum().item<float>() << std::endl;
    std::cout << "Number of non-zero elements (expected ~35): "
              << torch::count_nonzero(new_x).item<int>() << std::endl;
    std::cout << "A row from the new features:\n" << new_x[0] << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    FeatureDropout::FeatureDropout(double drop_rate, double p)
            : drop_rate_(drop_rate), p_(p) {

        if (drop_rate_ < 0.0 || drop_rate_ > 1.0) {
            throw std::invalid_argument("drop_rate must be in [0, 1].");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto FeatureDropout::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability Check ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("FeatureDropout expects 2 tensors: {node_features, edge_index}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || drop_rate_ == 0.0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        // --- 2. Apply Dropout ---
        // We can directly use the torch::nn::functional::dropout function.
        // It requires `is_training` to be true to have any effect.
        // It correctly handles the masking and scaling.
        torch::Tensor new_x = torch::nn::functional::dropout(
                x,
                torch::nn::functional::DropoutFuncOptions().p(drop_rate_).training(true)
        );

        // --- 3. Return the new graph structure ---
        return std::vector<torch::Tensor>{new_x, edge_index};
    }

} // namespace xt::transforms::graph