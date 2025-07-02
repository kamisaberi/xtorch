#include "include/losses/ghmr_loss.h"

namespace xt::losses
{
    /**
     * Gradient Harmonizing Mechanism for Regression (GHM-R) Loss function.
     * @param predictions Predicted values, shape: [batch_size]
     * @param targets Ground truth values, shape: [batch_size]
     * @param bins Number of bins for gradient density estimation, default: 10
     * @param momentum Momentum for updating gradient density, default: 0.9
     * @param beta Smooth L1 loss transition point, default: 0.1
     * @param eps Small value for numerical stability, default: 1e-6
     * @return Scalar tensor containing the GHM-R Loss
     */
    torch::Tensor ghmr_loss(const torch::Tensor& predictions, const torch::Tensor& targets, int bins = 10,
                            float momentum = 0.9,
                            float beta = 0.1,
                            float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = predictions.device();
        auto targets_device = targets.to(device);

        // Input validation
        TORCH_CHECK(predictions.dim() == 1, "Predictions must be 1D (batch_size)");
        TORCH_CHECK(targets.dim() == 1, "Targets must be 1D (batch_size)");
        TORCH_CHECK(predictions.size(0) == targets.size(0), "Batch sizes must match");
        TORCH_CHECK(bins > 0, "Number of bins must be positive");
        TORCH_CHECK(momentum >= 0 && momentum <= 1, "Momentum must be in [0, 1]");
        TORCH_CHECK(beta >= 0, "Beta must be non-negative");

        // Compute absolute error
        torch::Tensor abs_error = torch::abs(predictions - targets_device);

        // Compute gradient norm (approximated as absolute error for smooth L1 loss)
        torch::Tensor grad_norm = abs_error.clone();

        // Discretize gradient norms into bins
        torch::Tensor grad_min = grad_norm.min();
        torch::Tensor grad_max = grad_norm.max().clamp_max(1.0);
        torch::Tensor bin_width = (grad_max - grad_min) / bins;

        // Avoid division by zero in bin_width
        if (bin_width.item<float>() < eps)
        {
            bin_width = torch::tensor(eps, torch::TensorOptions().device(device));
        }

        // Compute bin indices
        torch::Tensor bin_indices = torch::floor((grad_norm - grad_min) / bin_width).to(torch::kLong);
        bin_indices = torch::clamp(bin_indices, 0, bins - 1);

        // Compute gradient density (number of samples per bin)
        torch::Tensor bin_counts = torch::zeros({bins}, torch::TensorOptions().device(device));
        bin_counts.scatter_add_(0, bin_indices, torch::ones_like(bin_indices, torch::kFloat));

        // Apply momentum to smooth bin counts (simulate running average)
        static torch::Tensor running_bin_counts = torch::zeros({bins}, torch::TensorOptions().device(device));
        if (running_bin_counts.device() != device)
        {
            running_bin_counts = torch::zeros({bins}, torch::TensorOptions().device(device));
        }
        running_bin_counts = momentum * running_bin_counts + (1.0 - momentum) * bin_counts;
        running_bin_counts = running_bin_counts.clamp_min(eps);

        // Compute weights as inverse of gradient density
        torch::Tensor weights = running_bin_counts.sum() / running_bin_counts.index({bin_indices});

        // Compute smooth L1 loss
        torch::Tensor loss = torch::where(
            abs_error < beta,
            0.5 * torch::pow(abs_error, 2) / beta,
            abs_error - 0.5 * beta
        );

        // Apply GHM weights
        loss = weights * loss;

        // Return mean loss over the batch
        return loss.mean();
    }

    auto GHMRLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::ghmr_loss(torch::zeros(10), torch::zeros(10));
    }
}
