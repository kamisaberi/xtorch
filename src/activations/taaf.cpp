#include "include/activations/taaf.h"

namespace xt::activations
{
    torch::Tensor taaf(
        const torch::Tensor& x,
        const torch::Tensor& alpha, // Per-channel learnable parameter
        double beta // Global hyperparameter or fixed
    )
    {
        // Assuming x is (Batch, Channels, ...) or (Batch, Features)
        // alpha should be (Channels) or (Features) to be broadcastable
        // For this simple function, alpha needs to be shaped to broadcast correctly.
        // E.g., if x is (N, C, H, W), alpha should be (1, C, 1, 1) or (C).
        // If x is (N, F), alpha should be (1, F) or (F).

        TORCH_CHECK(alpha.dim() > 0 && alpha.dim() <= x.dim(),
                    "alpha dimensions are not compatible for broadcasting with x.");
        // A more robust check would involve comparing specific dimensions if alpha is not scalar.
        // For example, if x is (N,C,H,W) and alpha is per-channel, alpha might be (C).
        // It would need to be reshaped to (1,C,1,1) for broadcasting.
        // For simplicity, we assume alpha is already shaped correctly for broadcasting or is scalar.

        torch::Tensor x_abs = torch::abs(x);
        torch::Tensor term_inside_log = 1.0 + torch::exp(-beta * (x_abs + alpha));
        torch::Tensor result = x / term_inside_log;

        return result;
    }

    auto TAAF::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::taaf(torch::zeros(10), torch::zeros(10));
    }
}
