#include "include/losses/dhel_loss.h"

namespace xt::losses
{
torch::Tensor dhel_loss(const torch::Tensor& features, const torch::Tensor& labels, float alpha = 1.0f, float beta = 0.1f, float gamma = 0.01f) {
    // Ensure inputs are valid
    TORCH_CHECK(features.dim() == 2, "Features must be 2D (batch_size, hash_dim)");
    TORCH_CHECK(labels.dim() == 2, "Labels must be 2D (batch_size, batch_size)");
    TORCH_CHECK(features.size(0) == labels.size(0), "Batch size mismatch between features and labels");
    TORCH_CHECK(labels.size(0) == labels.size(1), "Labels must be square (batch_size x batch_size)");
    TORCH_CHECK(features.dtype() == torch::kFloat, "Features must be float type");
    TORCH_CHECK(labels.dtype() == torch::kFloat, "Labels must be float type");
    TORCH_CHECK(alpha >= 0.0f, "Alpha must be non-negative");
    TORCH_CHECK(beta >= 0.0f, "Beta must be non-negative");
    TORCH_CHECK(gamma >= 0.0f, "Gamma must be non-negative");

    // Normalize features to [-1, 1] (assuming tanh activation)
    auto hash_codes = torch::tanh(features);

    // Pairwise similarity loss: encourage hash codes to be similar/dissimilar based on labels
    // Compute pairwise cosine similarity
    auto norm_codes = torch::nn::functional::normalize(hash_codes, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto similarity = torch::matmul(norm_codes, norm_codes.transpose(0, 1));

    // Target similarity: labels (1 for similar, 0 for dissimilar) scaled to [-1, 1]
    auto target_similarity = 2.0f * labels - 1.0f; // Map {0,1} to {-1,1}
    auto similarity_loss = torch::abs(similarity - target_similarity).mean();

    // Quantization loss: encourage hash codes to be binary-like (±1)
    auto quantization_loss = torch::abs(hash_codes - hash_codes.sign()).mean();

    // Bit balance loss: encourage each bit to be balanced (mean close to 0)
    auto bit_balance_loss = torch::abs(hash_codes.mean(0)).mean();

    // Combine losses with weights
    return alpha * similarity_loss + beta * quantization_loss + gamma * bit_balance_loss;
}

    auto DHELLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::dhel_loss(torch::zeros(10),torch::zeros(10));
    }
}
