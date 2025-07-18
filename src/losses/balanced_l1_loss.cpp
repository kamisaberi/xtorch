#include <losses/balanced_l1_loss.h>

namespace xt::losses
{

    torch::Tensor balanced_l1_loss(const torch::Tensor& input, const torch::Tensor& target, float alpha = 0.5f, float beta = 1.0f, float gamma = 1.5f) {
        // Ensure input and target have the same shape
        TORCH_CHECK(input.sizes() == target.sizes(), "Input and target must have the same shape");
        TORCH_CHECK(input.dtype() == torch::kFloat, "Input must be float type");
        TORCH_CHECK(target.dtype() == torch::kFloat, "Target must be float type");
        TORCH_CHECK(alpha >= 0.0f, "Alpha must be non-negative");
        TORCH_CHECK(beta > 0.0f, "Beta must be positive");
        TORCH_CHECK(gamma >= 0.0f, "Gamma must be non-negative");

        // Compute absolute difference
        auto diff = torch::abs(input - target);

        // Apply Balanced L1 Loss
        auto mask = diff < beta;
        auto loss_small = (alpha / beta) * diff * mask;
        auto loss_large = alpha * (diff - 0.5f * beta) * (1.0f - mask.to(torch::kFloat));

        // Combine losses
        auto loss = loss_small + loss_large;

        // Apply gamma scaling for additional balancing
        if (gamma != 0.0f) {
            loss = loss * torch::pow(diff / beta + 1.0f, gamma);
        }

        // Return mean loss
        return loss.mean();
    }
    auto BalancedL1Loss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::balanced_l1_loss(torch::zeros(10),torch::zeros(10));
    }
}
