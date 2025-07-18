#include <losses/zlpr_loss.h>

namespace xt::losses
{
    /**
     * Zero-bounded Log-sum-exp & Pairwise Rank-based (ZLPR) Loss function for ranking tasks.
     * Uses log-sum-exp to approximate maximum ranking violations and ensures zero loss for well-separated positive pairs.
     * @param embeddings Input embeddings, shape: [batch_size, embedding_dim]
     * @param labels Ground truth labels, shape: [batch_size]
     * @param margin Margin for negative pairs in ranking, default: 1.0
     * @param temperature Temperature parameter for log-sum-exp, default: 1.0
     * @param class_weights Weights for each class to handle imbalance, shape: [num_classes], default: nullptr (uniform)
     * @param eps Small value for numerical stability, default: 1e-6
     * @return Scalar tensor containing the ZLPR Loss
     */
    torch::Tensor zlpr_loss(const torch::Tensor& embeddings, const torch::Tensor& labels, float margin = 1.0,
                            float temperature = 1.0,
                            const torch::Tensor& class_weights = torch::Tensor(),
                            float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = embeddings.device();
        auto labels_device = labels.to(device).to(torch::kLong);

        // Input validation
        TORCH_CHECK(embeddings.dim() == 2, "Embeddings must be 2D (batch_size, embedding_dim)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(embeddings.size(0) == labels.size(0), "Batch sizes must match");
        TORCH_CHECK(margin >= 0, "Margin must be non-negative");
        TORCH_CHECK(temperature > 0, "Temperature must be positive");
        TORCH_CHECK((labels_device >= 0).all().item<bool>(), "Labels must be non-negative");
        if (class_weights.defined())
        {
            TORCH_CHECK(class_weights.dim() == 1, "Class weights must be 1D (num_classes)");
            TORCH_CHECK((class_weights >= 0).all().item<bool>(), "Class weights must be non-negative");
        }

        int batch_size = embeddings.size(0);

        // Normalize embeddings to unit vectors for cosine similarity
        torch::Tensor norm_emb = torch::nn::functional::normalize(
            embeddings, torch::nn::functional::NormalizeFuncOptions().dim(1).eps(eps));

        // Compute pairwise cosine similarities
        torch::Tensor similarities = torch::matmul(norm_emb, norm_emb.transpose(0, 1));
        // Shape: [batch_size, batch_size]

        // Create pairwise label match matrix (1 if same class, 0 otherwise)
        torch::Tensor label_matches = (labels_device.unsqueeze(1) == labels_device.unsqueeze(0)).to(torch::kFloat);

        // Mask to exclude self-comparisons (diagonal)
        torch::Tensor mask = torch::ones_like(label_matches) - torch::eye(batch_size, device = device);

        // Compute ranking violations for negative pairs: max(0, sim + margin - pos_sim)
        // For each anchor i, compute violations against all negative samples j
        torch::Tensor pos_sim = similarities * label_matches; // Positive pair similarities
        torch::Tensor neg_sim = similarities * (1.0 - label_matches); // Negative pair similarities
        torch::Tensor violations = torch::relu(neg_sim.unsqueeze(2) + margin - pos_sim.unsqueeze(1)) * mask.
            unsqueeze(2);

        // Apply log-sum-exp to approximate maximum violation per anchor
        torch::Tensor lse_violations = temperature * torch::logsumexp(violations / temperature, {1, 2});
        // Shape: [batch_size]

        // Zero-bounded: Apply ReLU to ensure loss is zero for anchors with no violations
        torch::Tensor loss = torch::relu(lse_violations);

        // Apply class weights
        torch::Tensor sample_weights = torch::ones({batch_size}, torch::TensorOptions().device(device));
        if (class_weights.defined())
        {
            sample_weights = class_weights.index({labels_device}).to(device);
        }

        // Compute weighted mean loss
        loss = (loss * sample_weights).sum() / sample_weights.sum().clamp_min(eps);

        return loss;
    }

    auto ZLPRLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::zlpr_loss(torch::zeros(10), torch::zeros(10));
    }
}
