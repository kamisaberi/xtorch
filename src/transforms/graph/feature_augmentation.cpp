#include "include/transforms/graph/feature_augmentation.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy graph.
    torch::Tensor x = torch::ones({5, 10}); // 5 nodes, 10 features
    torch::Tensor edge_index = torch::tensor({{0, 1}, {1, 0}}, torch::kLong);

    std::cout << "Mean of features before: " << x.mean().item<float>() << std::endl;
    std::cout << "Sum of features before: " << x.sum().item<float>() << std::endl;

    // --- Example 1: Noise Injection ---
    std::cout << "\n--- Applying Noise ---" << std::endl;
    xt::transforms::graph::FeatureAugmentation noise_adder("noise", 0.1);
    auto result_noise = noise_adder.forward({x, edge_index});
    auto graph_noise = std::any_cast<std::vector<torch::Tensor>>(result_noise);
    std::cout << "Mean of features after noise: " << graph_noise[0].mean().item<float>() << std::endl;

    // --- Example 2: Feature Masking ---
    std::cout << "\n--- Applying Masking ---" << std::endl;
    xt::transforms::graph::FeatureAugmentation masker("mask", 0.3);
    auto result_mask = masker.forward({x, edge_index});
    auto graph_mask = std::any_cast<std::vector<torch::Tensor>>(result_mask);
    // Expect 3 of 10 feature columns to be zeroed. Sum should be 5 * 10 * (1 - 0.3) = 35.
    std::cout << "Sum of features after mask: " << graph_mask[0].sum().item<float>() << std::endl;

    return 0;
}
*/

namespace xt::transforms::graph {

    FeatureAugmentation::FeatureAugmentation(const std::string& aug_type, double strength, double p)
            : strength_(strength), p_(p) {

        std::string aug_type_lower = aug_type;
        std::transform(aug_type_lower.begin(), aug_type_lower.end(), aug_type_lower.begin(), ::tolower);

        if (aug_type_lower == "noise") {
            aug_type_ = FeatureAugmentationType::Noise;
            if (strength_ < 0.0) throw std::invalid_argument("Strength for noise must be non-negative.");
        } else if (aug_type_lower == "mask") {
            aug_type_ = FeatureAugmentationType::Mask;
            if (strength_ < 0.0 || strength_ > 1.0) {
                throw std::invalid_argument("Strength for mask must be in [0, 1].");
            }
        } else {
            throw std::invalid_argument("Unknown augmentation type. Must be 'noise' or 'mask'.");
        }

        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be in [0, 1].");
        }
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto FeatureAugmentation::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation & Probability Check ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("FeatureAugmentation expects 2 tensors: {node_features, edge_index}.");
        }
        torch::Tensor x = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor edge_index = std::any_cast<torch::Tensor>(any_vec[1]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || strength_ == 0.0) {
            return std::vector<torch::Tensor>{x, edge_index};
        }

        auto new_x = x.clone();

        // --- 2. Apply Selected Augmentation ---
        if (aug_type_ == FeatureAugmentationType::Noise) {
            // Add Gaussian noise with a standard deviation of `strength_`.
            auto noise = torch::randn_like(x) * strength_;
            new_x += noise;

        } else if (aug_type_ == FeatureAugmentationType::Mask) {
            long num_features = x.size(1);
            if (num_features > 0) {
                // Create a mask for the feature dimensions (columns) to drop.
                auto drop_mask = torch::rand({num_features}, x.options()) < strength_;
                // auto drop_indices = std::get<0>(torch::where(drop_mask));
                auto drop_indices = torch::where(drop_mask)[0];


                if (drop_indices.numel() > 0) {
                    // Set the selected columns to zero for all nodes.
                    new_x.index_put_({"...", drop_indices}, 0.0);
                }
            }
        }

        // --- 3. Return the new graph structure ---
        return std::vector<torch::Tensor>{new_x, edge_index};
    }

} // namespace xt::transforms::graph