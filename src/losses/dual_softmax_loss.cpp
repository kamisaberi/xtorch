#include "include/losses/dual_softmax_loss.h"

namespace xt::losses
{
 torch::Tensor dual_softmax_loss(const torch::Tensor& features, const torch::Tensor& labels, const torch::Tensor& weights, float alpha = 1.0f, float beta = 1.0f, float temperature = 0.1f) {
    // Ensure inputs are valid
    TORCH_CHECK(features.dim() == 2, "Features must be 2D (batch_size, feature_dim)");
    TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
    TORCH_CHECK(weights.dim() == 2, "Weights must be 2D (num_classes, feature_dim)");
    TORCH_CHECK(features.size(0) == labels.size(0), "Batch size mismatch between features and labels");
    TORCH_CHECK(features.size(1) == weights.size(1), "Feature dimension mismatch between features and weights");
    TORCH_CHECK(features.dtype() == torch::kFloat, "Features must be float type");
    TORCH_CHECK(weights.dtype() == torch::kFloat, "Weights must be float type");
    TORCH_CHECK(labels.dtype() == torch::kLong, "Labels must be long type");
    TORCH_CHECK(alpha >= 0.0f, "Alpha must be non-negative");
    TORCH_CHECK(beta >= 0.0f, "Beta must be non-negative");
    TORCH_CHECK(temperature > 0.0f, "Temperature must be positive");

    // Normalize features and weights for cosine similarity
    auto norm_features = torch::nn::functional::normalize(features, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto norm_weights = torch::nn::functional::normalize(weights, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    // Classification Softmax Loss
    auto logits = torch::matmul(norm_features, norm_weights.transpose(0, 1));
    auto cls_loss = torch::nn::functional::cross_entropy(logits, labels);

    // Pairwise Softmax Loss
    // Compute pairwise cosine similarity
    auto similarity = torch::matmul(norm_features, norm_features.transpose(0, 1)) / temperature;

    // Create pairwise labels: 1 for same identity, 0 for different
    auto batch_size = labels.size(0);
    auto label_matrix = labels.view({-1, 1}).eq(labels.view({1, -1})).to(torch::kFloat);

    // Apply softmax to similarity scores and compute cross-entropy
    auto pair_logits = similarity - (1.0f - label_matrix) * 1e10f; // Mask out negative pairs for numerical stability
    auto pair_loss = -torch::log_softmax(pair_logits, 1) * label_matrix;
    pair_loss = pair_loss.sum() / (label_matrix.sum() + 1e-6f); // Normalize by number of positive pairs

    // Combine losses
    return alpha * cls_loss + beta * pair_loss;
}

    auto DualSoftmaxLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::dual_softmax_loss(torch::zeros(10));
    }
}
