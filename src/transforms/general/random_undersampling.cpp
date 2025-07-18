#include <transforms/general/random_undersampling.h>

namespace xt::transforms::general {

    RandomUnderSampling::RandomUnderSampling() = default;

    auto RandomUnderSampling::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Casting ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("RandomUnderSampling::forward expects exactly 2 tensors: features and labels.");
        }

        torch::Tensor X = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor y = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!X.defined() || !y.defined()) {
            throw std::invalid_argument("Input features or labels are not defined.");
        }
        if (X.dim() != 2) {
            throw std::invalid_argument("Features tensor (X) must be 2D [n_samples, n_features].");
        }
        if (y.dim() != 1) {
            throw std::invalid_argument("Labels tensor (y) must be 1D [n_samples].");
        }
        if (X.size(0) != y.size(0)) {
            throw std::invalid_argument("Number of samples in features and labels must match.");
        }

        // 2. --- Calculate Class Distribution ---
        auto y_cpu = y.to(torch::kCPU);
        auto y_accessor = y_cpu.accessor<int64_t, 1>();

        std::map<int64_t, int64_t> class_counts;
        int64_t minority_count = std::numeric_limits<int64_t>::max();

        for (int64_t i = 0; i < y_accessor.size(0); ++i) {
            class_counts[y_accessor[i]]++;
        }

        for (const auto& pair : class_counts) {
            if (pair.second < minority_count) {
                minority_count = pair.second;
            }
        }

        // If all classes have the same number of samples, no need to do anything.
        // We can just return the original shuffled data.
        bool is_balanced = true;
        for (const auto& pair : class_counts) {
            if(pair.second != minority_count) {
                is_balanced = false;
                break;
            }
        }
        if (is_balanced) {
            torch::Tensor shuffle_indices = torch::randperm(X.size(0), X.options().dtype(torch::kLong));
            return std::make_pair(X.index_select(0, shuffle_indices), y.index_select(0, shuffle_indices));
        }

        // 3. --- Perform Under-sampling ---
        std::vector<torch::Tensor> resampled_x_list;
        std::vector<torch::Tensor> resampled_y_list;

        for (const auto& pair : class_counts) {
            int64_t class_label = pair.first;

            // Get indices of all samples belonging to the current class
            torch::Tensor class_indices = (y == class_label).nonzero().squeeze(-1);

            // Randomly select `minority_count` indices from this class's samples
            // torch::randperm is perfect for creating random selections without replacement.
            torch::Tensor permuted_class_indices = class_indices.index_select(0, torch::randperm(class_indices.size(0)));
            torch::Tensor selected_indices = permuted_class_indices.slice(/*dim=*/0, /*start=*/0, /*end=*/minority_count);

            // Add the randomly selected samples to our list
            resampled_x_list.push_back(X.index_select(0, selected_indices));
            resampled_y_list.push_back(y.index_select(0, selected_indices));
        }

        // 4. --- Concatenate and Shuffle the Final Dataset ---
        torch::Tensor resampled_X = torch::cat(resampled_x_list, 0);
        torch::Tensor resampled_y = torch::cat(resampled_y_list, 0);

        // Shuffle the final combined dataset to ensure classes are mixed
        torch::Tensor shuffle_indices = torch::randperm(resampled_X.size(0), X.options().dtype(torch::kLong));

        resampled_X = resampled_X.index_select(0, shuffle_indices);
        resampled_y = resampled_y.index_select(0, shuffle_indices);

        // 5. --- Return the resampled features and labels ---
        return std::make_pair(resampled_X, resampled_y);
    }

} // namespace xt::transforms::general