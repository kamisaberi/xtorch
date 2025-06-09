#include "include/losses/nt_xent.h"

namespace xt::losses
{
    torch::Tensor nt_xent_loss(const torch::Tensor& features1, const torch::Tensor& features2, float temperature = 0.5f)
    {
        // Ensure inputs are valid
        TORCH_CHECK(features1.dim() == 2, "Features1 must be 2D (batch_size, feature_dim)");
        TORCH_CHECK(features2.dim() == 2, "Features2 must be 2D (batch_size, feature_dim)");
        TORCH_CHECK(features1.size(0) == features2.size(0), "Batch sizes must match");
        TORCH_CHECK(features1.size(1) == features2.size(1), "Feature dimensions must match");
        TORCH_CHECK(features1.dtype() == torch::kFloat, "Features1 must be float type");
        TORCH_CHECK(features2.dtype() == torch::kFloat, "Features2 must be float type");
        TORCH_CHECK(temperature > 0.0f, "Temperature must be positive");

        // Normalize features for cosine similarity
        auto norm_features1 = torch::nn::functional::normalize(
            features1, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        auto norm_features2 = torch::nn::functional::normalize(
            features2, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

        // Concatenate features for pairwise similarity
        auto batch_size = features1.size(0);
        auto all_features = torch::cat({norm_features1, norm_features2}, 0); // Shape: (2*batch_size, feature_dim)

        // Compute cosine similarity matrix
        auto similarity = torch::matmul(all_features, all_features.transpose(0, 1)) / temperature;
        // Shape: (2*batch_size, 2*batch_size)

        // Create labels for positive pairs
        auto labels = torch::arange(batch_size, torch::kLong);
        auto positive_mask = torch::zeros({2 * batch_size, 2 * batch_size}, torch::kFloat);
        for (int64_t i = 0; i < batch_size; ++i)
        {
            positive_mask.index_put_({i, batch_size + i}, 1.0f); // features1[i] -> features2[i]
            positive_mask.index_put_({batch_size + i, i}, 1.0f); // features2[i] -> features1[i]
        }

        // Compute NT-Xent loss
        auto max_sim = std::get<0>(torch::max(similarity, 1, true)); // Extract max values from tuple
        auto logits = similarity - max_sim; // Subtract max for numerical stability
        auto exp_logits = torch::exp(logits);
        auto log_prob = logits - torch::log(exp_logits.sum(1, true) + 1e-10f);
        auto loss = -torch::sum(log_prob * positive_mask) / (positive_mask.sum() + 1e-10f);

        return loss;
    }

    auto NTXent::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::nt_xent(torch::zeros(10));
    }
}
