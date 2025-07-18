#include <losses/varifocal_loss.h>

namespace xt::losses
{
    /**
 * Varifocal Loss function for dense object detection.
 * Balances positive and negative samples using IoU-aware weights and focal loss.
 * @param logits Input classification logits, shape: [batch_size, num_anchors, num_classes]
 * @param targets Ground truth labels (0 for background, 1 to num_classes-1 for foreground), shape: [batch_size, num_anchors]
 * @param iou_scores IoU scores for positive samples, shape: [batch_size, num_anchors]
 * @param alpha Focal loss parameter for negative samples, default: 0.75
 * @param gamma Focal loss focusing parameter, default: 2.0
 * @param eps Small value for numerical stability, default: 1e-6
 * @return Scalar tensor containing the Varifocal Loss
 */
    torch::Tensor varifocal_loss(const torch::Tensor& logits, const torch::Tensor& targets,
                                 const torch::Tensor& iou_scores,
                                 float alpha = 0.75,
                                 float gamma = 2.0,
                                 float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = logits.device();
        auto targets_device = targets.to(device).to(torch::kLong);
        auto iou_scores_device = iou_scores.to(device);

        // Input validation
        TORCH_CHECK(logits.dim() == 3, "Logits must be 3D (batch_size, num_anchors, num_classes)");
        TORCH_CHECK(targets.dim() == 2, "Targets must be 2D (batch_size, num_anchors)");
        TORCH_CHECK(iou_scores.dim() == 2, "IoU scores must be 2D (batch_size, num_anchors)");
        TORCH_CHECK(logits.size(0) == targets.size(0) && logits.size(1) == targets.size(1),
                    "Batch size and anchor dimensions must match");
        TORCH_CHECK(iou_scores.size(0) == targets.size(0) && iou_scores.size(1) == targets.size(1),
                    "IoU scores dimensions must match targets");
        TORCH_CHECK((targets_device >= 0).all().item<bool>(), "Targets must be non-negative");
        TORCH_CHECK((iou_scores_device >= 0).all().item<bool>() && (iou_scores_device <= 1).all().item<bool>(),
                    "IoU scores must be in [0, 1]");
        TORCH_CHECK(alpha >= 0 && alpha <= 1, "Alpha must be in [0, 1]");
        TORCH_CHECK(gamma >= 0, "Gamma must be non-negative");

        int batch_size = logits.size(0);
        int num_anchors = logits.size(1);
        int num_classes = logits.size(2);

        // Compute sigmoid probabilities
        torch::Tensor probs = torch::sigmoid(logits);

        // Create binary targets for each class (1 for positive, 0 for negative or background)
        torch::Tensor pos_mask = (targets_device > 0).to(torch::kFloat); // Shape: [batch_size, num_anchors]

        // Create one-hot targets for positive samples only
        torch::Tensor one_hot_targets = torch::zeros({batch_size, num_anchors, num_classes},
                                                     torch::TensorOptions().dtype(torch::kFloat).device(device));
        torch::Tensor indices = targets_device.unsqueeze(2); // Shape: [batch_size, num_anchors, 1]
        one_hot_targets.scatter_(
            2, indices, pos_mask.unsqueeze(2) * torch::tensor(1.0, torch::TensorOptions().device(device)));

        // Get IoU-aware targets: use iou_scores for positive samples, 0 for negatives/background
        torch::Tensor iou_targets = pos_mask * iou_scores_device; // Shape: [batch_size, num_anchors]

        // Compute binary cross-entropy loss
        torch::Tensor bce_loss = -iou_targets.unsqueeze(2) * torch::log(probs.clamp(eps, 1.0 - eps)) -
            (1.0 - iou_targets.unsqueeze(2)) * torch::log((1.0 - probs).clamp(eps, 1.0 - eps));

        // Compute focal weights: (1-p)^gamma for positives, alpha*p^gamma for negatives
        torch::Tensor focal_weight = torch::where(
            one_hot_targets > 0,
            torch::pow(1.0 - probs, gamma) * iou_targets.unsqueeze(2),
            alpha * torch::pow(probs, gamma)
        );

        // Compute weighted loss
        torch::Tensor loss = focal_weight * bce_loss;

        // Sum over anchors and classes, average over batch
        torch::Tensor valid_mask = (targets_device >= 0).to(torch::kFloat); // Exclude invalid targets if any
        torch::Tensor loss_sum = loss.sum({1, 2}); // Sum over anchors and classes
        torch::Tensor valid_count = valid_mask.sum().clamp_min(eps);
        loss = loss_sum.sum() / valid_count;

        return loss;
    }

    auto VarifocalLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::varifocal_loss(torch::zeros(10), torch::zeros(10), torch::zeros(10));
    }
}
