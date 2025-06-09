#include "include/losses/dice_bce_loss.h"

namespace xt::losses
{

    torch::Tensor dice_bce_loss(const torch::Tensor& input, const torch::Tensor& target, float smooth = 1.0f) {
        // Ensure input and target have the same shape
        TORCH_CHECK(input.sizes() == target.sizes(), "Input and target must have the same shape");
        TORCH_CHECK(input.dtype() == torch::kFloat, "Input must be float type");
        TORCH_CHECK(target.dtype() == torch::kFloat, "Target must be float type");

        // Flatten input and target for Dice computation
        auto input_flat = input.view({-1});
        auto target_flat = target.view({-1});

        // Compute intersection
        auto intersection = (input_flat * target_flat).sum();

        // Compute Dice coefficient
        auto dice = (2.0f * intersection + smooth) /
                   (input_flat.sum() + target_flat.sum() + smooth);

        // Return Dice Loss (1 - Dice coefficient)
        return 1.0f - dice;
    }


    auto DiceBCELoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::dice_bce_loss(torch::zeros(10) , torch::zeros(10));
    }
}
