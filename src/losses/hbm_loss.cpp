#include <losses/hbm_loss.h>

namespace xt::losses
{
    /**
   * Hierarchy-aware Biased Bound Margin (HBBM) Loss function for multi-class classification.
   * Incorporates hierarchical class weights, biased weighting for class imbalance, and bounded margins.
   * @param logits Input logits, shape: [batch_size, num_classes]
   * @param labels Ground truth labels, shape: [batch_size]
   * @param hierarchy_weights Hierarchical class importance weights, shape: [num_classes], default: nullptr (uniform)
   * @param margin_max Maximum margin for separating target and non-target classes, default: 0.5
   * @param bias_factor Factor to upweight minority classes, default: 1.0
   * @param eps Small value for numerical stability, default: 1e-6
   * @return Scalar tensor containing the HBBM Loss
   */
    torch::Tensor hbm_loss(const torch::Tensor& logits, const torch::Tensor& labels,
                           const torch::Tensor& hierarchy_weights = torch::Tensor(), float margin_max = 0.5,
                           float bias_factor = 1.0,
                           float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = logits.device();
        auto labels_device = labels.to(device);

        // Input validation
        TORCH_CHECK(logits.dim() == 2, "Logits must be 2D (batch_size, num_classes)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(logits.size(0) == labels.size(0), "Batch sizes must match");
        TORCH_CHECK(margin_max >= 0, "Maximum margin must be non-negative");
        TORCH_CHECK(bias_factor >= 0, "Bias factor must be non-negative");
        if (hierarchy_weights.defined())
        {
            TORCH_CHECK(hierarchy_weights.dim() == 1 && hierarchy_weights.size(0) == logits.size(1),
                        "Hierarchy weights must be 1D with size equal to num_classes");
            TORCH_CHECK((hierarchy_weights >= 0).all().item<bool>(), "Hierarchy weights must be non-negative");
        }

        // Compute softmax probabilities for bias weighting
        torch::Tensor probs = torch::softmax(logits, 1);

        // Create one-hot encoding of labels
        auto one_hot = torch::zeros_like(logits).scatter_(1, labels_device.view({-1, 1}), 1.0);

        // Get probabilities for target classes
        torch::Tensor target_probs = (probs * one_hot).sum(1).clamp(eps, 1.0 - eps);

        // Compute bounded margin: scale margin based on target probability to bound its effect
        torch::Tensor margins = margin_max * (1.0 - target_probs);
        margins = margins.clamp(0.0, margin_max); // Ensure margin is bounded

        // Apply margin to target logits
        torch::Tensor modified_logits = logits.clone();
        modified_logits = modified_logits + margins.unsqueeze(1) * one_hot - margins.unsqueeze(1) * (1.0 - one_hot);

        // Compute cross-entropy loss
        torch::Tensor ce_loss = -torch::log_softmax(modified_logits, 1) * one_hot;
        ce_loss = ce_loss.sum(1);

        // Apply hierarchical weights
        torch::Tensor class_weights;
        if (hierarchy_weights.defined())
        {
            class_weights = hierarchy_weights.index({labels_device}).to(device);
        }
        else
        {
            class_weights = torch::ones({labels.size(0)}, torch::TensorOptions().device(device));
        }

        // Compute biased weights to emphasize minority classes
        torch::Tensor class_counts = torch::zeros({logits.size(1)}, torch::TensorOptions().device(device));
        class_counts.scatter_add_(0, labels_device, torch::ones_like(labels_device, torch::kFloat));
        class_counts = class_counts.clamp_min(eps);
        torch::Tensor bias_weights = torch::pow(logits.size(0) / class_counts.index({labels_device}), bias_factor);
        bias_weights = bias_weights.clamp_max(10.0); // Prevent extreme weights

        // Combine weights and loss
        torch::Tensor loss = class_weights * bias_weights * ce_loss;

        // Return mean loss over the batch
        return loss.mean();
    }

    auto HBMLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::hbm_loss(torch::zeros(10), torch::zeros(10));
    }
}
