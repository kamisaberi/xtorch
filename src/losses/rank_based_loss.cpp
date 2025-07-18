#include <losses/rank_based_loss.h>

namespace xt::losses
{
    torch::Tensor rank_based_loss(const torch::Tensor& features, const torch::Tensor& labels, float margin = 1.0f,
                                  float temperature = 0.1f)
    {
        // Ensure inputs are valid
        TORCH_CHECK(features.dim() == 2, "Features must be 2D (batch_size, feature_dim)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(features.size(0) == labels.size(0), "Batch size mismatch between features and labels");
        TORCH_CHECK(features.dtype() == torch::kFloat, "Features must be float type");
        TORCH_CHECK(labels.dtype() == torch::kFloat, "Labels must be float type");
        TORCH_CHECK(margin >= 0.0f, "Margin must be non-negative");
        TORCH_CHECK(temperature > 0.0f, "Temperature must be positive");

        // Normalize features for cosine similarity
        auto norm_features = torch::nn::functional::normalize(
            features, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

        // Compute pairwise cosine similarity
        auto similarity = torch::matmul(norm_features, norm_features.transpose(0, 1)) / temperature;
        // Shape: (batch_size, batch_size)

        // Compute label differences to determine correct ranking
        auto label_diff = labels.view({-1, 1}) - labels.view({1, -1}); // Shape: (batch_size, batch_size)
        auto pos_mask = label_diff > 0; // Positive pairs: i should rank higher than j
        auto neg_mask = label_diff < 0; // Negative pairs: j should rank higher than i

        // Compute pairwise ranking loss
        auto pos_loss = torch::relu(margin - similarity) * pos_mask.to(torch::kFloat);
        auto neg_loss = torch::relu(similarity + margin) * neg_mask.to(torch::kFloat);

        // Combine and average loss
        auto loss = (pos_loss + neg_loss).sum() / (pos_mask.sum() + neg_mask.sum() + 1e-6f);

        return loss;
    }

    auto RankBasedLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::rank_based_loss(torch::zeros(10),torch::zeros(10));
    }
}
