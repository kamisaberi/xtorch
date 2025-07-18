#include <transforms/general/random_oversampling.h>
#include <stdexcept>
#include <map>

namespace xt::transforms::general {

    RandomOverSampling::RandomOverSampling() = default;

    auto RandomOverSampling::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Casting ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("RandomOverSampling::forward expects exactly 2 tensors: features and labels.");
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
        // Move labels to CPU to iterate over them easily
        auto y_cpu = y.to(torch::kCPU);
        auto y_accessor = y_cpu.accessor<int64_t, 1>();

        std::map<int64_t, int64_t> class_counts;
        int64_t majority_count = 0;
        int64_t majority_class = -1;

        for (int64_t i = 0; i < y_accessor.size(0); ++i) {
            class_counts[y_accessor[i]]++;
        }

        for (const auto& pair : class_counts) {
            if (pair.second > majority_count) {
                majority_count = pair.second;
                majority_class = pair.first;
            }
        }

        // 3. --- Perform Oversampling ---
        std::vector<torch::Tensor> resampled_x_list;
        std::vector<torch::Tensor> resampled_y_list;

        for (const auto& pair : class_counts) {
            int64_t class_label = pair.first;
            int64_t num_samples = pair.second;

            // Get indices of all samples belonging to the current class
            torch::Tensor class_indices = (y == class_label).nonzero().squeeze(-1);
            // Get the actual feature data for this class
            torch::Tensor x_class = X.index_select(0, class_indices);

            // Add the original samples for this class to our list
            resampled_x_list.push_back(x_class);
            resampled_y_list.push_back(y.index_select(0, class_indices));

            if (class_label != majority_class) {
                // This is a minority class, we need to add more samples
                int64_t num_to_add = majority_count - num_samples;
                if (num_to_add > 0) {
                    // Generate random indices to duplicate from this class's samples
                    torch::Tensor random_indices = torch::randint(0, num_samples, {num_to_add}, X.options().dtype(torch::kLong));

                    // Add the duplicated samples
                    resampled_x_list.push_back(x_class.index_select(0, random_indices));
                    resampled_y_list.push_back(y.index_select(0, class_indices).index_select(0, random_indices));
                }
            }
        }

        // 4. --- Concatenate and Shuffle the Final Dataset ---
        torch::Tensor resampled_X = torch::cat(resampled_x_list, 0);
        torch::Tensor resampled_y = torch::cat(resampled_y_list, 0);

        // It's crucial to shuffle the resampled dataset to break up the blocks
        // of duplicated data.
        torch::Tensor shuffle_indices = torch::randperm(resampled_X.size(0), X.options().dtype(torch::kLong));

        resampled_X = resampled_X.index_select(0, shuffle_indices);
        resampled_y = resampled_y.index_select(0, shuffle_indices);

        // 5. --- Return the resampled features and labels ---
        // We return a std::pair wrapped in std::any.
        return std::make_pair(resampled_X, resampled_y);
    }

} // namespace xt::transforms::general