#include "include/losses/adaptive_loss.h"

namespace xt::losses
{
    torch::Tensor adaptive_loss(const torch::Tensor& input, const torch::Tensor& target, float smooth = 1.0f, float alpha = 0.5f) {
        // Ensure input and target have the same shape
        TORCH_CHECK(input.sizes() == target.sizes(), "Input and target must have the same shape");
        TORCH_CHECK(input.dtype() == torch::kFloat, "Input must be float type");
        TORCH_CHECK(target.dtype() == torch::kFloat, "Target must be float type");

        // Compute BCE Loss
        auto bce_loss = torch::binary_cross_entropy(input, target);

        // Flatten input and target for Dice computation
        auto input_flat = input.view({-1});
        auto target_flat = target.view({-1});

        // Compute intersection for Dice
        auto intersection = (input_flat * target_flat).sum();

        // Compute Dice coefficient
        auto dice = (2.0f * intersection + smooth) /
                   (input_flat.sum() + target_flat.sum() + smooth);

        // Compute Dice Loss
        auto dice_loss = 1.0f - dice;

        // Adaptive weighting based on loss magnitudes
        auto bce_magnitude = bce_loss.item<float>();
        auto dice_magnitude = dice_loss.item<float>();
        auto total_magnitude = bce_magnitude + dice_magnitude + 1e-6f; // Avoid division by zero
        auto bce_weight = dice_magnitude / total_magnitude; // Inverse weighting
        auto dice_weight = bce_magnitude / total_magnitude;

        // Apply momentum to weights for stability
        bce_weight = alpha * bce_weight + (1.0f - alpha) * (bce_magnitude > dice_magnitude ? 0.6f : 0.4f);
        dice_weight = alpha * dice_weight + (1.0f - alpha) * (dice_magnitude > bce_magnitude ? 0.6f : 0.4f);

        // Normalize weights
        auto weight_sum = bce_weight + dice_weight;
        bce_weight /= weight_sum;
        dice_weight /= weight_sum;

        // Combine losses with adaptive weights
        return bce_weight * bce_loss + dice_weight * dice_loss;
    }
    auto AdaptiveLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::adaptive_loss(torch::zeros(10),torch::zeros(10));
    }
}
