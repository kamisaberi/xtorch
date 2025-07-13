#include "include/transforms/target/label_balancer.h"
#include <stdexcept>
#include <chrono>
#include <unordered_map>

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

// Helper to count labels in a tensor
void print_label_counts(const torch::Tensor& labels) {
    auto unique_labels = std::get<0>(torch::_unique(labels));
    std::cout << "Label Counts: { ";
    for (int i = 0; i < unique_labels.size(0); ++i) {
        long label_val = unique_labels[i].item<long>();
        long count = (labels == label_val).sum().item<long>();
        std::cout << label_val << ": " << count << " ";
    }
    std::cout << "}" << std::endl;
}

int main() {
    // 1. --- Setup: Create an imbalanced dataset ---
    // 10 samples of class 0, and only 3 samples of class 1.
    torch::Tensor features = torch::cat({
        torch::randn({10, 4}),      // 10 samples for class 0
        torch::randn({3, 4}) + 5.0  // 3 samples for class 1 (add 5 to make them distinct)
    });
    torch::Tensor labels = torch::cat({
        torch::zeros({10}, torch::kLong),
        torch::ones({3}, torch::kLong)
    });

    std::cout << "--- Original Dataset ---" << std::endl;
    std::cout << "Total Samples: " << labels.size(0) << std::endl;
    print_label_counts(labels);

    // --- Example 1: Oversampling ---
    // Should bring class 1 up to 10 samples, for a total of 20.
    std::cout << "\n--- Testing OVERSAMPLING ---" << std::endl;
    xt::transforms::target::LabelBalancer oversampler(xt::transforms::target::BalancingStrategy::OVERSAMPLE);
    auto over_any = oversampler.forward({features, labels});
    auto over_pair = std::any_cast<std::pair<torch::Tensor, torch::Tensor>>(over_any);

    std::cout << "New Total Samples: " << over_pair.second.size(0) << std::endl;
    print_label_counts(over_pair.second);

    // --- Example 2: Undersampling ---
    // Should bring class 0 down to 3 samples, for a total of 6.
    std::cout << "\n--- Testing UNDERSAMPLING ---" << std::endl;
    xt::transforms::target::LabelBalancer undersampler(xt::transforms::target::BalancingStrategy::UNDERSAMPLE);
    auto under_any = undersampler.forward({features, labels});
    auto under_pair = std::any_cast<std::pair<torch::Tensor, torch::Tensor>>(under_any);

    std::cout << "New Total Samples: " << under_pair.second.size(0) << std::endl;
    print_label_counts(under_pair.second);

    return 0;
}
*/

namespace xt::transforms::target {

    LabelBalancer::LabelBalancer(BalancingStrategy strategy) : strategy_(strategy) {
        unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        random_engine_.seed(seed);
    }

    auto LabelBalancer::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("LabelBalancer::forward requires {features_tensor, labels_tensor}.");
        }

        torch::Tensor features, labels;
        try {
            features = std::any_cast<torch::Tensor>(any_vec[0]);
            labels = std::any_cast<torch::Tensor>(any_vec[1]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Inputs to LabelBalancer must be torch::Tensors.");
        }
        if (features.size(0) != labels.size(0)) {
            throw std::invalid_argument("Features and labels must have the same number of samples.");
        }
        if (labels.dim() != 1) {
            throw std::invalid_argument("Labels tensor must be 1-dimensional.");
        }
        if (features.device() != labels.device()) {
             throw std::invalid_argument("Features and labels must be on the same device.");
        }

        // 2. --- Get Class Counts and Determine Target Size ---
        auto unique_labels = std::get<0>(torch::_unique(labels));
        if (unique_labels.numel() <= 1) return std::make_pair(features, labels); // Already balanced or empty

        std::vector<long> counts_vec;
        for(int i = 0; i < unique_labels.size(0); ++i) {
            counts_vec.push_back((labels == unique_labels[i]).sum().item<long>());
        }
        torch::Tensor counts = torch::tensor(counts_vec, labels.options());

        long target_sample_count = 0;
        if (strategy_ == BalancingStrategy::OVERSAMPLE) {
            // CORRECTED LINE: Call .item<long>() directly on the result of .max()
            target_sample_count = counts.max().item<long>();
        } else { // UNDERSAMPLE
            // CORRECTED LINE: Call .item<long>() directly on the result of .min()
            target_sample_count = counts.min().item<long>();
        }

        // 3. --- Resample Each Class ---
        std::vector<torch::Tensor> balanced_feature_batches;
        std::vector<torch::Tensor> balanced_label_batches;

        for (int i = 0; i < unique_labels.size(0); ++i) {
            long class_id = unique_labels[i].item<long>();
            torch::Tensor class_indices = (labels == class_id).nonzero().squeeze(-1);
            long current_count = class_indices.size(0);

            if (current_count == 0) continue;

            torch::Tensor resample_indices = torch::randint(0, current_count, {target_sample_count}, labels.options());
            torch::Tensor final_indices_for_class = class_indices.index({resample_indices});

            balanced_feature_batches.push_back(features.index({final_indices_for_class}));
            balanced_label_batches.push_back(labels.index({final_indices_for_class}));
        }

        // 4. --- Combine and Shuffle ---
        if (balanced_feature_batches.empty()) {
            return std::make_pair(torch::empty({0}, features.options()), torch::empty({0}, labels.options()));
        }
        torch::Tensor balanced_features = torch::cat(balanced_feature_batches, 0);
        torch::Tensor balanced_labels = torch::cat(balanced_label_batches, 0);

        torch::Tensor shuffle_perm = torch::randperm(balanced_labels.size(0), balanced_labels.options());

        balanced_features = balanced_features.index({shuffle_perm});
        balanced_labels = balanced_labels.index({shuffle_perm});

        return std::make_pair(balanced_features, balanced_labels);
    }

} // namespace xt::transforms::target```