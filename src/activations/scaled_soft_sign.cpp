#include <activations/scaled_soft_sign.h>

namespace xt::activations
{
    torch::Tensor scaled_soft_sign(const torch::Tensor& x, double scale_in, double scale_out)
    {
        torch::Tensor scaled_x = scale_in * x;
        torch::Tensor soft_sign_x = scaled_x / (1.0 + torch::abs(scaled_x));
        return scale_out * soft_sign_x;
    }

    auto ScaledSoftSign::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::scaled_soft_sign(torch::zeros(10));
    }
}
