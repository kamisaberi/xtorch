#include "include/losses/info_nce_loss.h"

namespace xt::losses
{
    /**
     * InfoNCE Loss function for contrastive learning.
     * Encourages positive pairs to have high similarity and negative pairs to have low similarity.
     * @param embeddings1 First set of embeddings, shape: [batch_size, embedding_dim]
     * @param embeddings2 Second set of embeddings (e.g., augmentations), shape: [batch_size, embedding_dim]
     * @param temperature Temperature parameter for scaling similarities, default: 0.1
     * @param eps Small value for numerical stability, default: 1e-6
     * @return Scalar tensor containing the InfoNCE Loss
     */
    torch::Tensor info_nce_loss(const torch::Tensor& embeddings1, const torch::Tensor& embeddings2,
                                float temperature = 0.1, float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = embeddings1.device();
        TORCH_CHECK(embeddings2.device() == device, "Embeddings must be on the same device");

        // Input validation
        TORCH_CHECK(embeddings1.dim() == 2, "embeddings1 must be 2D (batch_size, embedding_dim)");
        TORCH_CHECK(embeddings2.dim() == 2, "embeddings2 must be 2D (batch_size, embedding_dim)");
        TORCH_CHECK(embeddings1.size(0) == embeddings2.size(0), "Batch sizes must match");
        TORCH_CHECK(embeddings1.size(1) == embeddings2.size(1), "Embedding dimensions must match");
        TORCH_CHECK(temperature > 0, "Temperature must be positive");

        int batch_size = embeddings1.size(0);

        // Normalize embeddings to unit vectors for cosine similarity
        torch::Tensor norm_emb1 = torch::nn::functional::normalize(embeddings1,
                                                                   torch::nn::functional::NormalizeFuncOptions().dim(1).
                                                                   eps(eps));
        torch::Tensor norm_emb2 = torch::nn::functional::normalize(embeddings2,
                                                                   torch::nn::functional::NormalizeFuncOptions().dim(1).
                                                                   eps(eps));

        // Compute similarity matrix (cosine similarity)
        torch::Tensor similarities = torch::matmul(norm_emb1, norm_emb2.transpose(0, 1)) / temperature;

        // Create labels for positive pairs (diagonal elements, i.e., each sample with itself)
        torch::Tensor labels = torch::arange(batch_size, torch::TensorOptions().dtype(torch::kLong).device(device)).
            unsqueeze(1);

        // Compute log-softmax over similarities
        torch::Tensor log_probs = torch::nn::functional::log_softmax(similarities, 1);

        // Extract log probabilities for positive pairs
        torch::Tensor pos_log_probs = log_probs.diagonal();

        // Compute InfoNCE loss (negative log-likelihood of positive pairs)
        torch::Tensor loss = -pos_log_probs.mean();

        return loss;
    }

    auto InfoNCELoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::info_nce_loss(torch::zeros(10), torch::zeros(10));
    }
}
