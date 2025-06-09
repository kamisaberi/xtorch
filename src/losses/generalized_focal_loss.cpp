#include "include/losses/generalized_focal_loss.h"

namespace xt::losses
{
    /**
     * Generalized Focal Loss function for multi-class classification with class imbalance.
     * @param logits Input logits, shape: [batch_size, num_classes]
     * @param labels Ground truth labels, shape: [batch_size]
     * @param alpha Class weights to handle imbalance, shape: [num_classes], default: nullptr (uniform weights)
     * @param gamma Focusing parameter to emphasize hard examples, default: 2.0
     * @param eps Small value for numerical stability, default: 1e-6
     * @return Scalar tensor containing the Generalized Focal Loss
     */
    torch::Tensor generalized_focal_loss(const torch::Tensor& logits, const torch::Tensor& labels,
                                         const torch::Tensor& alpha = torch::Tensor(), float gamma = 2.0,
                                         float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = logits.device();
        auto labels_device = labels.to(device);

        // Input validation
        TORCH_CHECK(logits.dim() == 2, "Logits must be 2D (batch_size, num_classes)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(logits.size(0) == labels.size(0), "Batch sizes must match");
        TORCH_CHECK(gamma >= 0, "Gamma must be non-negative");
        if (alpha.defined())
        {
            TORCH_CHECK(alpha.dim() == 1 && alpha.size(0) == logits.size(1),
                        "Alpha must be 1D with size equal to num_classes");
            TORCH_CHECK((alpha >= 0).all().item<bool>(), "Alpha values must be non-negative");
        }

        // Compute softmax probabilities
        torch::Tensor probs = torch::softmax(logits, 1);

        // Create one-hot encoding of labels
        auto one_hot = torch::zeros_like(logits).scatter_(1, labels_device.view({-1, 1}), 1.0);

        // Get probabilities for target classes
        torch::Tensor target_probs = (probs * one_hot).sum(1).clamp(eps, 1.0 - eps);

        // Compute focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        torch::Tensor focal_factor = torch::pow(1.0 - target_probs, gamma);
        torch::Tensor log_probs = torch::log(target_probs);

        // Apply alpha weighting
        torch::Tensor alpha_weights;
        if (alpha.defined())
        {
            alpha_weights = alpha.index({labels_device}).to(device);
        }
        else
        {
            alpha_weights = torch::ones({labels.size(0)}, device = device);
        }

        // Compute loss
        torch::Tensor loss = -alpha_weights * focal_factor * log_probs;

        // Return mean loss over the batch
        return loss.mean();
    }

    auto GeneralizedFocalLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::generalized_focal_loss(torch::zeros(10), torch::zeros(10));
    }
}
