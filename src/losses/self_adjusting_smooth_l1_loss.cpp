#include <losses/self_adjusting_smooth_l1_loss.h>

namespace xt::losses
{
    torch::Tensor self_adjusting_smooth_l1_loss(const torch::Tensor& input, const torch::Tensor& target,
                                                float alpha = 1.0f, float percentile = 0.5f, float momentum = 0.1f)
    {
        // Ensure inputs are valid
        TORCH_CHECK(input.sizes() == target.sizes(), "Input and target must have the same shape");
        TORCH_CHECK(input.dtype() == torch::kFloat, "Input must be float type");
        TORCH_CHECK(target.dtype() == torch::kFloat, "Target must be float type");
        TORCH_CHECK(alpha >= 0.0f, "Alpha must be non-negative");
        TORCH_CHECK(percentile > 0.0f && percentile < 1.0f, "Percentile must be in (0, 1)");
        TORCH_CHECK(momentum >= 0.0f && momentum <= 1.0f, "Momentum must be in [0, 1]");

        // Compute absolute difference
        auto diff = torch::abs(input - target);

        // Compute beta based on the percentile of absolute differences
        auto flat_diff = diff.view(-1);
        auto sort_result = torch::sort(flat_diff); // Returns tuple of (values, indices)
        auto sorted_diff = std::get<0>(sort_result); // Extract sorted values
        auto index = static_cast<int64_t>(percentile * (sorted_diff.size(0) - 1));
        auto batch_beta = sorted_diff.index({index}).item<float>();
        batch_beta = std::max(batch_beta, 1e-6f); // Ensure beta is positive

        // Static variable to maintain running average of beta
        static float running_beta = 1.0f; // Initial beta
        running_beta = momentum * batch_beta + (1.0f - momentum) * running_beta;

        // Compute Smooth L1 Loss with adjusted beta
        auto beta = running_beta;
        auto mask = diff < beta;
        auto loss_small = 0.5f * diff * diff / beta * mask;
        auto loss_large = (diff - 0.5f * beta) * (1.0f - mask.to(torch::kFloat));

        // Combine losses
        auto loss = loss_small + loss_large;

        // Apply scaling factor
        return alpha * loss.mean();
    }

    auto SelfAdjustingSmoothL1Loss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::self_adjusting_smooth_l1_loss(torch::zeros(10),torch::zeros(10));
    }
}
