#include "include/losses/triplet_loss.h"

namespace xt::losses
{
    torch::Tensor triplet_loss(const torch::Tensor& features, const torch::Tensor& labels, float margin = 1.0f)
    {
        // Ensure inputs are valid
        TORCH_CHECK(features.dim() == 2, "Features must be 2D (batch_size, feature_dim)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(features.size(0) == labels.size(0), "Batch size mismatch between features and labels");
        TORCH_CHECK(features.dtype() == torch::kFloat, "Features must be float type");
        TORCH_CHECK(labels.dtype() == torch::kLong, "Labels must be long type");
        TORCH_CHECK(margin >= 0.0f, "Margin must be non-negative");

        auto batch_size = features.size(0);

        // Normalize features for cosine similarity
        auto norm_features = torch::nn::functional::normalize(
            features, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

        // Compute pairwise cosine similarity
        auto similarity = torch::matmul(norm_features, norm_features.transpose(0, 1));
        // Shape: (batch_size, batch_size)

        // Initialize triplet loss
        torch::Tensor triplet_loss = torch::zeros({}, torch::kFloat);
        int64_t valid_triplets = 0;

        // Form triplets and compute triplet loss
        for (int64_t anchor = 0; anchor < batch_size; ++anchor)
        {
            auto anchor_label = labels[anchor];

            // Find one positive and one negative
            int64_t positive = -1, negative = -1;
            for (int64_t j = 0; j < batch_size; ++j)
            {
                if (j != anchor)
                {
                    if (labels[j].item<int64_t>() == anchor_label.item<int64_t>() && positive == -1)
                    {
                        positive = j; // First positive
                    }
                    else if (labels[j].item<int64_t>() != anchor_label.item<int64_t>() && negative == -1)
                    {
                        negative = j; // First negative
                    }
                    if (positive != -1 && negative != -1) break;
                }
            }

            // Compute triplet loss if valid triplet found
            if (positive != -1 && negative != -1)
            {
                auto pos_sim = similarity[anchor][positive];
                auto neg_sim = similarity[anchor][negative];
                triplet_loss += torch::relu(margin - pos_sim + neg_sim);
                valid_triplets++;
            }
        }

        // Average triplet loss
        if (valid_triplets > 0)
        {
            triplet_loss /= valid_triplets;
        }
        else
        {
            triplet_loss = torch::tensor(0.0f, torch::kFloat);
        }

        return triplet_loss;
    }

    auto TripletLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::triplet_loss(torch::zeros(10));
    }
}
