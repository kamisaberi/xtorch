#include <losses/focal_loss.h>

namespace xt::losses
{
    /**
     * Focal Loss function for classification tasks, addressing class imbalance.
     * @param logits Input logits, shape: [batch_size, num_classes]
     * @param labels Ground truth labels, shape: [batch_size]
     * @param alpha Weighting factor for class imbalance (default: 0.25)
     * @param gamma Focusing parameter to emphasize hard examples (default: 2.0)
     * @param eps Small value for numerical stability (default: 1e-6)
     * @return Scalar tensor containing the Focal Loss
     */
    torch::Tensor focal_loss(const torch::Tensor& logits, const torch::Tensor& labels, float alpha = 0.25,
                             float gamma = 2.0, float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = logits.device();
        auto labels_device = labels.to(device);

        // Input validation
        TORCH_CHECK(logits.dim() == 2, "Logits must be 2D (batch_size, num_classes)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(logits.size(0) == labels.size(0), "Batch sizes must match");
        TORCH_CHECK(alpha >= 0 && alpha <= 1, "Alpha must be in [0, 1]");
        TORCH_CHECK(gamma >= 0, "Gamma must be non-negative");

        // Compute softmax probabilities
        torch::Tensor probs = torch::softmax(logits, 1);

        // Create one-hot encoding of labels
        auto one_hot = torch::zeros_like(logits).scatter_(1, labels_device.view({-1, 1}), 1.0);

        // Get probabilities for the target classes
        torch::Tensor target_probs = (probs * one_hot).sum(1).clamp(eps, 1.0 - eps);

        // Compute focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
        torch::Tensor loss = -alpha * torch::pow(1.0 - target_probs, gamma) * torch::log(target_probs);

        // Return mean loss over the batch
        return loss.mean();
    }

    auto FocalLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::focal_loss(torch::zeros(10), torch::zeros(10));
    }
}
