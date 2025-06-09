#include "include/losses/ghmc_loss.h"

namespace xt::losses
{
/**
 * Gradient Harmonizing Mechanism (GHM) Loss function for multi-class classification.
 * @param logits Input logits, shape: [batch_size, num_classes]
 * @param labels Ground truth labels, shape: [batch_size]
 * @param bins Number of bins for gradient density estimation, default: 10
 * @param momentum Momentum for updating gradient density, default: 0.9
 * @param eps Small value for numerical stability, default: 1e-6
 * @return Scalar tensor containing the GHM-C Loss
 */
torch::Tensor ghmc_loss(
    const torch::Tensor& logits,
    const torch::Tensor& labels,
    int bins = 10,
    float momentum = 0.9,
    float eps = 1e-6) {
    // Ensure inputs are on the same device
    auto device = logits.device();
    auto labels_device = labels.to(device);

    // Input validation
    TORCH_CHECK(logits.dim() == 2, "Logits must be 2D (batch_size, num_classes)");
    TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
    TORCH_CHECK(logits.size(0) == labels.size(0), "Batch sizes must match");
    TORCH_CHECK(bins > 0, "Number of bins must be positive");
    TORCH_CHECK(momentum >= 0 && momentum <= 1, "Momentum must be in [0, 1]");

    // Compute softmax probabilities
    torch::Tensor probs = torch::softmax(logits, 1);

    // Create one-hot encoding of labels
    auto one_hot = torch::zeros_like(logits).scatter_(1, labels_device.view({-1, 1}), 1.0);

    // Get probabilities for target classes
    torch::Tensor target_probs = (probs * one_hot).sum(1).clamp(eps, 1.0 - eps);

    // Compute gradient norm (approximated as p_t for cross-entropy)
    torch::Tensor grad_norm = target_probs;

    // Discretize gradient norms into bins
    torch::Tensor grad_min = grad_norm.min();
    torch::Tensor grad_max = grad_norm.max().clamp_max(1.0 - eps);
    torch::Tensor bin_width = (grad_max - grad_min) / bins;

    // Avoid division by zero in bin_width
    if (bin_width.item<float>() < eps) {
        bin_width = torch::tensor(eps, device=device);
    }

    // Compute bin indices
    torch::Tensor bin_indices = torch::floor((grad_norm - grad_min) / bin_width).to(torch::kLong);
    bin_indices = torch::clamp(bin_indices, 0, bins - 1);

    // Compute gradient density (number of samples per bin)
    torch::Tensor bin_counts = torch::zeros({bins}, device=device);
    bin_counts.scatter_add_(0, bin_indices, torch::ones_like(bin_indices, torch::kFloat));

    // Apply momentum to smooth bin counts (simulate running average)
    static torch::Tensor running_bin_counts = torch::zeros({bins}, device=device);
    if (running_bin_counts.device() != device) {
        running_bin_counts = torch::zeros({bins}, device=device);
    }
    running_bin_counts = momentum * running_bin_counts + (1.0 - momentum) * bin_counts;
    running_bin_counts = running_bin_counts.clamp_min(eps);

    // Compute weights as inverse of gradient density
    torch::Tensor weights = running_bin_counts.sum() / running_bin_counts.index({bin_indices});

    // Compute cross-entropy loss
    torch::Tensor ce_loss = -torch::log(target_probs);

    // Apply GHM weights
    torch::Tensor loss = weights * ce_loss;

    // Return mean loss over the batch
    return loss.mean();
}
    auto GHMCLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::ghmc(torch::zeros(10));
    }
}
