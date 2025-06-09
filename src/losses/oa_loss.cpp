#include "include/losses/oa_loss.h"

namespace xt::losses
{
    torch::Tensor object_aware_loss(const torch::Tensor& pred, const torch::Tensor& target,
                                    const torch::Tensor& object_mask, float dice_weight = 1.0f,
                                    float aware_weight = 0.1f, float smooth = 1.0f)
    {
        // Ensure inputs are valid
        TORCH_CHECK(pred.sizes() == target.sizes(), "Prediction and target must have the same shape");
        TORCH_CHECK(pred.sizes() == object_mask.sizes(), "Object mask must match prediction and target shapes");
        TORCH_CHECK(pred.dtype() == torch::kFloat, "Prediction must be float type");
        TORCH_CHECK(target.dtype() == torch::kFloat, "Target must be float type");
        TORCH_CHECK(object_mask.dtype() == torch::kFloat, "Object mask must be float type");
        TORCH_CHECK(dice_weight >= 0.0f, "Dice weight must be non-negative");
        TORCH_CHECK(aware_weight >= 0.0f, "Object-aware weight must be non-negative");
        TORCH_CHECK(smooth >= 0.0f, "Smooth must be non-negative");

        // Dice Loss
        auto pred_flat = pred.view(-1);
        auto target_flat = target.view(-1);
        auto intersection = (pred_flat * target_flat).sum();
        auto dice = (2.0f * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth);
        auto dice_loss = 1.0f - dice;

        // Object-Aware Regularization: Penalize predictions in non-object regions
        // Assumes object_mask is binary or soft [0,1], penalizes pred where mask is 0
        auto aware_loss = torch::abs(pred * (1.0f - object_mask)).mean();

        // Combine losses
        return dice_weight * dice_loss + aware_weight * aware_loss;
    }

    auto ObjectAwareLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::object_aware_loss(torch::zeros(10));
    }
}
