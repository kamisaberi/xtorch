#include "include/losses/cycle_consistency_loss.h"

namespace xt::losses
{
    torch::Tensor cycle_consistency_loss(const torch::Tensor& original, const torch::Tensor& reconstructed, float weight = 1.0f) {
        // Ensure original and reconstructed have the same shape
        TORCH_CHECK(original.sizes() == reconstructed.sizes(), "Original and reconstructed must have the same shape");
        TORCH_CHECK(original.dtype() == torch::kFloat, "Original must be float type");
        TORCH_CHECK(reconstructed.dtype() == torch::kFloat, "Reconstructed must be float type");
        TORCH_CHECK(weight >= 0.0f, "Weight must be non-negative");

        // Compute L1 loss (mean absolute error) between original and reconstructed
        auto diff = torch::abs(original - reconstructed);
        auto loss = diff.mean();

        // Apply weight
        return weight * loss;
    }
    auto CycleConsistencyLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::cycle_consistency_loss(torch::zeros(10),torch::zeros(10));
    }
}
