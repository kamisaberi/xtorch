#include "include/losses/dynamic_smooth_l1_loss.h"

namespace xt::losses
{

    torch::Tensor dynamic_smooth_l1_loss(const torch::Tensor& input, const torch::Tensor& target, float alpha = 1.0f, float percentile = 0.5f) {
        // Ensure inputs are valid
        TORCH_CHECK(input.sizes() == target.sizes(), "Input and target must have the same shape");
        TORCH_CHECK(input.dtype() == torch::kFloat, "Input must be float type");
        TORCH_CHECK(target.dtype() == torch::kFloat, "Target must be float type");
        TORCH_CHECK(alpha >= 0.0f, "Alpha must be non-negative");
        TORCH_CHECK(percentile > 0.0f && percentile < 1.0f, "Percentile must be in (0, 1)");

        // Compute absolute difference
        auto diff = torch::abs(input - target);

        // Dynamically compute beta based on the percentile of absolute differences
        auto flat_diff = diff.view(-1);
        auto sort_result = torch::sort(flat_diff); // Returns tuple of (values, indices)
        auto sorted_diff = std::get<0>(sort_result); // Extract sorted values
        auto index = static_cast<int64_t>(percentile * (sorted_diff.size(0) - 1));
        auto beta = sorted_diff.index({index}).item<float>();
        beta = std::max(beta, 1e-6f); // Ensure beta is positive to avoid division by zero

        // Compute Smooth L1 Loss
        auto mask = diff < beta;
        auto loss_small = 0.5f * diff * diff / beta * mask;
        auto loss_large = (diff - 0.5f * beta) * (1.0f - mask.to(torch::kFloat));

        // Combine losses
        auto loss = loss_small + loss_large;

        // Apply scaling factor
        return alpha * loss.mean();
    }

    auto DynamicSmoothL1Loss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::dynamic_smooth_l1_loss(torch::zeros(10),torch::zeros(10));
    }
}
