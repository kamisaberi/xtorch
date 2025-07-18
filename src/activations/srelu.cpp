#include <activations/srelu.h>

namespace xt::activations
{
    torch::Tensor srelu(
        const torch::Tensor& x,
        const torch::Tensor& t_left, // Threshold for left part
        const torch::Tensor& a_left, // Slope for left part
        const torch::Tensor& t_right, // Threshold for right part
        const torch::Tensor& a_right // Slope for right part
    )
    {
        // Parameters t_left, a_left, t_right, a_right can be scalars
        // or tensors for per-channel/element adaptivity.
        // For this simple function, we assume they are broadcastable to x.
        // The paper ensures t_left <= t_right.
        // For a simple functional version, we won't enforce this constraint here,
        // but it's crucial for the intended SReLU shape.
        // If t_left > t_right, the behavior is undefined by the paper's description.

        TORCH_CHECK(
            t_left.numel() == 1 || t_left.sizes() == x.sizes() || (t_left.dim() <= x.dim() /* for broadcasting */),
            "t_left must be scalar, same shape as x, or broadcastable.");
        TORCH_CHECK(a_left.numel() == 1 || a_left.sizes() == x.sizes() || (a_left.dim() <= x.dim()),
                    "a_left must be scalar, same shape as x, or broadcastable.");
        TORCH_CHECK(t_right.numel() == 1 || t_right.sizes() == x.sizes() || (t_right.dim() <= x.dim()),
                    "t_right must be scalar, same shape as x, or broadcastable.");
        TORCH_CHECK(a_right.numel() == 1 || a_right.sizes() == x.sizes() || (a_right.dim() <= x.dim()),
                    "a_right must be scalar, same shape as x, or broadcastable.");

        // y_l(x) = t_l + a_l * (x - t_l)
        // y_r(x) = t_r + a_r * (x - t_r)
        // f(x)   = x

        torch::Tensor y_l = t_left + a_left * (x - t_left);
        torch::Tensor y_r = t_right + a_right * (x - t_right);

        torch::Tensor result = torch::where(
            x <= t_left,
            y_l,
            torch::where(
                x >= t_right,
                y_r,
                x // Middle part is identity
            )
        );

        return result;
    }

    auto SReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::srelu(torch::zeros(10), torch::zeros(10), torch::zeros(10), torch::zeros(10),
                                      torch::zeros(10));
    }
}
