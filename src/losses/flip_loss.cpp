#include <losses/flip_loss.h>

namespace xt::losses
{
    /**
     * FlipLoss function for classification tasks with random logit flipping for robustness.
     * @param logits Input logits, shape: [batch_size, num_classes]
     * @param labels Ground truth labels, shape: [batch_size]
     * @param flip_prob Probability of flipping logits for non-target classes
     * @param s Scaling factor for logits
     * @param eps Small value for numerical stability
     * @return Scalar tensor containing the FlipLoss
     */
    torch::Tensor flip_loss(const torch::Tensor& logits, const torch::Tensor& labels, float flip_prob, float s,
                            float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = logits.device();
        auto labels_device = labels.to(device);

        // Input validation
        TORCH_CHECK(logits.dim() == 2, "Logits must be 2D (batch_size, num_classes)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(logits.size(0) == labels.size(0), "Batch sizes must match");
        TORCH_CHECK(flip_prob >= 0 && flip_prob <= 1, "Flip probability must be in [0, 1]");
        TORCH_CHECK(s >= 0, "Scaling factor must be non-negative");

        // Create one-hot encoding of labels
        auto one_hot = torch::zeros_like(logits).scatter_(1, labels_device.view({-1, 1}), 1.0);

        // Generate random mask for flipping logits (0 or 1 based on flip_prob)
        torch::Tensor flip_mask = torch::rand_like(logits, device = device) < flip_prob;
        flip_mask = flip_mask * (1.0 - one_hot); // Only flip non-target class logits

        // Flip logits for non-target classes where flip_mask is 1
        torch::Tensor modified_logits = logits.clone();
        modified_logits = torch::where(flip_mask, -modified_logits, modified_logits);

        // Apply scaling
        modified_logits = s * modified_logits;

        // Compute cross-entropy loss
        torch::Tensor loss = torch::nn::functional::cross_entropy(modified_logits, labels_device);
        return loss;
    }

    auto FLIPLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::flip_loss(torch::zeros(10), torch::zeros(10), 0.0f, 0.0f);
    }
}
