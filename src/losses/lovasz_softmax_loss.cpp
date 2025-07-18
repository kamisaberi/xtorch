#include <losses/lovasz_softmax_loss.h>

namespace xt::losses
{
    /**
     * Lovász-Softmax Loss function for multi-class segmentation tasks.
     * Optimizes mean Intersection over Union (mIoU) using a differentiable Jaccard loss surrogate.
     * @param logits Input logits, shape: [batch_size, num_classes, height, width]
     * @param labels Ground truth labels, shape: [batch_size, height, width]
     * @param class_weights Weights for each class to handle imbalance, shape: [num_classes], default: nullptr (uniform)
     * @param ignore_index Label value to ignore (e.g., void class), default: -1
     * @param eps Small value for numerical stability, default: 1e-6
     * @return Scalar tensor containing the Lovász-Softmax Loss
     */
    torch::Tensor lovasz_softmax_loss(const torch::Tensor& logits, const torch::Tensor& labels,
                                      const torch::Tensor& class_weights = torch::Tensor(),
                                      int ignore_index = -1,
                                      float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = logits.device();
        auto labels_device = labels.to(device);

        // Input validation
        TORCH_CHECK(logits.dim() == 4, "Logits must be 4D (batch_size, num_classes, height, width)");
        TORCH_CHECK(labels.dim() == 3, "Labels must be 3D (batch_size, height, width)");
        TORCH_CHECK(logits.size(0) == labels.size(0), "Batch sizes must match");
        TORCH_CHECK(logits.size(2) == labels.size(1) && logits.size(3) == labels.size(2),
                    "Spatial dimensions must match");
        if (class_weights.defined())
        {
            TORCH_CHECK(class_weights.dim() == 1 && class_weights.size(0) == logits.size(1),
                        "Class weights must be 1D with size equal to num_classes");
            TORCH_CHECK((class_weights >= 0).all().item<bool>(), "Class weights must be non-negative");
        }

        int batch_size = logits.size(0);
        const int num_classes = logits.size(1);
        int height = logits.size(2);
        int width = logits.size(3);

        // Compute softmax probabilities
        torch::Tensor probs = torch::softmax(logits, 1);

        // Initialize loss
        torch::Tensor loss = torch::zeros(1, torch::TensorOptions().device(device));

        // Process each class
        for (int c = 0; c < num_classes; ++c)
        {
            // Create binary ground truth for class c (1 if label == c, 0 otherwise)
            torch::Tensor gt_c = (labels_device == c).to(torch::kFloat);

            // Apply ignore mask
            if (ignore_index != -1)
            {
                gt_c = gt_c * (labels_device != ignore_index).to(torch::kFloat);
            }

            // Get probabilities for class c
            torch::Tensor prob_c = probs.select(1, c); // Shape: [batch_size, height, width]

            // Flatten for Lovász computation
            torch::Tensor prob_flat = prob_c.view({batch_size, -1}); // Shape: [batch_size, height*width]
            torch::Tensor gt_flat = gt_c.view({batch_size, -1}); // Shape: [batch_size, height*width]

            // Compute errors (1 - p for positives, p for negatives)
            torch::Tensor errors = torch::where(gt_flat > 0.5, 1.0 - prob_flat, prob_flat);

            // Sort errors in descending order
            auto [sorted_errors, indices] = errors.sort(1, true); // Shape: [batch_size, height*width]
            torch::Tensor sorted_gt = gt_flat.gather(1, indices); // Align ground truth with sorted errors

            // Compute Lovász gradient (Jaccard loss gradient)
            torch::Tensor cumsum_gt = sorted_gt.cumsum(1); // Cumulative sum of ground truth
            torch::Tensor cumsum_neg = (1.0 - sorted_gt).cumsum(1); // Cumulative sum of negatives
            torch::Tensor iou_grad = cumsum_gt / (cumsum_gt + cumsum_neg).clamp_min(eps); // IoU gradient

            // Compute Lovász delta (differences in IoU gradient)
            torch::Tensor delta = iou_grad.diff(1, 1); // Differences along dim 1, shape: [batch_size, height*width-1]
            // Pad delta with the first IoU gradient to match sorted_errors length
            torch::Tensor first_grad = iou_grad.select(1, 0).unsqueeze(1); // Shape: [batch_size, 1]
            delta = torch::cat({first_grad, delta}, 1); // Shape: [batch_size, height*width]

            // Compute Lovász loss for class c
            torch::Tensor loss_c = (sorted_errors * delta).sum(1).mean(); // Mean over batch

            // Apply class weight
            float weight = class_weights.defined() ? class_weights[c].item<float>() : 1.0;
            loss += weight * loss_c;
        }

        // Normalize by number of classes (if no weights provided)
        if (!class_weights.defined())
        {
            loss /= num_classes;
        }

        return loss;
    }

    auto LovaszSoftmaxLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::lovasz_softmax_loss(torch::zeros(10), torch::zeros(10));
    }
}
