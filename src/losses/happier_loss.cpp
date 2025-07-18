#include <losses/happier_loss.h>

namespace xt::losses
{
    /**
     * Hierarchical Average Precision training for Pertinent ImagE Retrieval (HAPPIER) Loss function.
     * Designed for image retrieval tasks to optimize ranking with hierarchical class weighting.
     * @param similarities Pairwise similarity scores, shape: [batch_size, batch_size]
     * @param labels Class labels for images, shape: [batch_size]
     * @param hierarchy_weights Hierarchical class importance weights, shape: [num_classes], default: nullptr (uniform)
     * @param tau Temperature parameter for sigmoid approximation, default: 1.0
     * @param eps Small value for numerical stability, default: 1e-6
     * @return Scalar tensor containing the HAPPIER Loss
     */
    torch::Tensor happier_loss(const torch::Tensor& similarities, const torch::Tensor& labels,
                               const torch::Tensor& hierarchy_weights = torch::Tensor(),
                               float tau = 1.0,
                               float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = similarities.device();
        auto labels_device = labels.to(device);

        // Input validation
        TORCH_CHECK(similarities.dim() == 2 && similarities.size(0) == similarities.size(1),
                    "Similarities must be 2D square matrix (batch_size, batch_size)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(similarities.size(0) == labels.size(0), "Batch size mismatch");
        TORCH_CHECK(tau > 0, "Tau must be positive");
        if (hierarchy_weights.defined())
        {
            TORCH_CHECK(hierarchy_weights.dim() == 1, "Hierarchy weights must be 1D (num_classes)");
            TORCH_CHECK((hierarchy_weights >= 0).all().item<bool>(), "Hierarchy weights must be non-negative");
        }

        int batch_size = similarities.size(0);

        // Create pairwise label match matrix (1 if same class, 0 otherwise)
        torch::Tensor label_matches = (labels_device.unsqueeze(1) == labels_device.unsqueeze(0)).to(torch::kFloat);

        // Apply hierarchical weights to positive pairs
        torch::Tensor pair_weights = torch::ones_like(label_matches);
        if (hierarchy_weights.defined())
        {
            torch::Tensor sample_weights = hierarchy_weights.index({labels_device}).to(device);
            pair_weights = sample_weights.unsqueeze(1) * sample_weights.unsqueeze(0);
            pair_weights = pair_weights * label_matches; // Only apply to positive pairs
        }

        // Compute ranking scores using sigmoid approximation for AP
        torch::Tensor rank_scores = torch::sigmoid(similarities / tau);

        // Mask to exclude self-comparisons (diagonal)
        torch::Tensor mask = torch::ones_like(similarities) - torch::eye(
            batch_size, torch::TensorOptions().device(device));

        // Compute positive and negative contributions
        torch::Tensor pos_scores = rank_scores * label_matches * mask;
        torch::Tensor neg_scores = rank_scores * (1.0 - label_matches) * mask;

        // Approximate AP loss: encourage positive scores to be higher than negative scores
        torch::Tensor diff = pos_scores.unsqueeze(2) - neg_scores.unsqueeze(1);
        torch::Tensor ap_loss = torch::sigmoid(-diff / tau).clamp(eps, 1.0 - eps); // Smooth ranking loss

        // Weight loss by hierarchical importance
        torch::Tensor weighted_loss = ap_loss * pair_weights.unsqueeze(2);

        // Compute mean loss, considering only valid pairs
        torch::Tensor valid_mask = label_matches.unsqueeze(2) * (1.0 - label_matches).unsqueeze(1) * mask.unsqueeze(2);
        torch::Tensor loss = weighted_loss.sum() / valid_mask.sum().clamp_min(eps);

        return loss;
    }

    auto HAPPIERLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::happier_loss(torch::zeros(10), torch::zeros(10));
    }
}
