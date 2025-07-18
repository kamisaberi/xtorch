#include <losses/arcface_loss.h>

namespace xt::losses
{
    torch::Tensor arcface_loss(const torch::Tensor& features, const torch::Tensor& labels, const torch::Tensor& weights,
                               float s = 30.0f, float m = 0.5f)
    {
        // Ensure inputs are valid
        TORCH_CHECK(features.dim() == 2, "Features must be 2D (batch_size, feature_dim)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(weights.dim() == 2, "Weights must be 2D (num_classes, feature_dim)");
        TORCH_CHECK(features.size(0) == labels.size(0), "Batch size mismatch between features and labels");
        TORCH_CHECK(features.size(1) == weights.size(1), "Feature dimension mismatch between features and weights");
        TORCH_CHECK(features.dtype() == torch::kFloat, "Features must be float type");
        TORCH_CHECK(weights.dtype() == torch::kFloat, "Weights must be float type");
        TORCH_CHECK(labels.dtype() == torch::kLong, "Labels must be long type");

        // Check if labels are within bounds
        auto num_classes = weights.size(0);
        TORCH_CHECK(labels.max().item<int64_t>() < num_classes, "Label indices exceed number of classes");
        TORCH_CHECK(labels.min().item<int64_t>() >= 0, "Label indices must be non-negative");

        // Normalize features and weights
        auto norm_features = torch::nn::functional::normalize(
            features, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        auto norm_weights = torch::nn::functional::normalize(
            weights, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

        // Compute cosine similarity (logits)
        auto cos_theta = torch::matmul(norm_features, norm_weights.transpose(0, 1));

        // Get target cosine values
        auto batch_size = features.size(0);
        auto target_cos = cos_theta.index_select(1, labels).diagonal();

        // Compute arcface margin: cos(theta + m)
        auto theta = torch::acos(target_cos.clamp(-1.0f + 1e-7f, 1.0f - 1e-7f));
        auto target_cos_m = torch::cos(theta + m);

        // Create one-hot mask for target classes
        auto one_hot = torch::zeros_like(cos_theta, torch::kFloat);
        one_hot.scatter_(1, labels.view({-1, 1}), torch::tensor(1.0f, torch::kFloat));

        // Apply margin only to target classes
        auto logits = cos_theta + one_hot * (target_cos_m - target_cos);

        // Scale logits
        logits = logits * s;

        // Compute cross-entropy loss
        return torch::nn::functional::cross_entropy(logits, labels);
    }

    auto ArcFaceLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::arcface_loss(torch::zeros(10), torch::zeros(10), torch::zeros(10));
    }
}
