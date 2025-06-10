#include "include/activations/margin_relu.h"

namespace xt::activations
{
    torch::Tensor margin_relu(const torch::Tensor& x, double margin_neg, double margin_pos)
    {
        TORCH_CHECK(margin_neg <= margin_pos, "margin_neg must be less than or equal to margin_pos.");

        torch::Tensor result = torch::where(
            x < margin_neg,
            torch::tensor(0.0, x.options()),
            torch::where(
                x > margin_pos,
                torch::tensor(1.0, x.options()),
                (x - margin_neg) / (margin_pos - margin_neg + 1e-7)
                // Add epsilon for numerical stability if margin_pos == margin_neg
            )
        );
        return result;
    }

    auto MarginReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::margin_relu(torch::zeros(10));
    }
}
