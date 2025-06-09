#include "include/losses/dsam_loss.h"

namespace xt::losses
{
    torch::Tensor dsam_loss(const torch::Tensor& pred, const torch::Tensor& target, const torch::Tensor& attention_map, float dice_weight = 1.0f, float attention_weight = 0.1f, float smooth = 1.0f) {
        // Ensure inputs are valid
        TORCH_CHECK(pred.sizes() == target.sizes(), "Prediction and target must have the same shape");
        TORCH_CHECK(pred.sizes() == attention_map.sizes(), "Attention map must match prediction and target shapes");
        TORCH_CHECK(pred.dtype() == torch::kFloat, "Prediction must be float type");
        TORCH_CHECK(target.dtype() == torch::kFloat, "Target must be float type");
        TORCH_CHECK(attention_map.dtype() == torch::kFloat, "Attention map must be float type");
        TORCH_CHECK(dice_weight >= 0.0f, "Dice weight must be non-negative");
        TORCH_CHECK(attention_weight >= 0.0f, "Attention weight must be non-negative");
        TORCH_CHECK(smooth >= 0.0f, "Smooth must be non-negative");

        // Dice Loss
        auto pred_flat = pred.view({-1});
        auto target_flat = target.view({-1});
        auto intersection = (pred_flat * target_flat).sum();
        auto dice = (2.0f * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth);
        auto dice_loss = 1.0f - dice;

        // Attention Regularization Loss: Encourage attention map to align with target
        // Assumes attention_map is normalized (e.g., via sigmoid, values in [0,1])
        auto attention_loss = torch::abs(attention_map - target).mean();

        // Combine losses
        return dice_weight * dice_loss + attention_weight * attention_loss;
    }

    auto DSAMLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::dsam_loss(torch::zeros(10),torch::zeros(10),torch::zeros(10));
    }
}
