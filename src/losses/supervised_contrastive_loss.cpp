#include <losses/supervised_contrastive_loss.h>

namespace xt::losses
{
    torch::Tensor supervised_contrastive_loss(const torch::Tensor& features, const torch::Tensor& labels,
                                              float temperature = 0.1f)
    {
        // Ensure inputs are valid
        TORCH_CHECK(features.dim() == 2, "Features must be 2D (batch_size, feature_dim)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(features.size(0) == labels.size(0), "Batch size mismatch between features and labels");
        TORCH_CHECK(features.dtype() == torch::kFloat, "Features must be float type");
        TORCH_CHECK(labels.dtype() == torch::kLong, "Labels must be long type");
        TORCH_CHECK(temperature > 0.0f, "Temperature must be positive");

        // Normalize features for cosine similarity
        auto norm_features = torch::nn::functional::normalize(
            features, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

        // Compute pairwise cosine similarity
        auto similarity = torch::matmul(norm_features, norm_features.transpose(0, 1)) / temperature;
        // Shape: (batch_size, batch_size)

        // Create positive pair mask (same labels)
        auto batch_size = features.size(0);
        auto pos_mask = labels.view({-1, 1}).eq(labels.view({1, -1})).to(torch::kFloat);
        pos_mask = pos_mask * (1.0f - torch::eye(batch_size, torch::kFloat)); // Exclude self-pairs by zeroing diagonal

        // Compute log-sum-exp for the denominator
        auto exp_sim = torch::exp(similarity);
        auto sum_exp_sim = exp_sim.sum(1, true) - exp_sim.diagonal().view({-1, 1}); // Exclude self-similarity
        auto log_sum_exp = torch::log(sum_exp_sim + 1e-10f);

        // Compute loss for each sample
        auto log_prob = similarity - log_sum_exp;
        auto pos_log_prob = log_prob * pos_mask;
        auto num_positives = pos_mask.sum(1, true).clamp(1.0f); // Avoid division by zero
        auto loss = -torch::sum(pos_log_prob, 1) / num_positives;

        // Average loss over valid samples
        return loss.mean();
    }

    auto SupervisedContrastiveLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::supervised_contrastive_loss(torch::zeros(10), torch::zeros(10));
    }
}
