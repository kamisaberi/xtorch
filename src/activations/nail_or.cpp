#include <activations/nail_or.h>

namespace xt::activations
{
    torch::Tensor nail_or(const torch::Tensor& x, const torch::Tensor& z)
    {
        TORCH_CHECK(x.sizes() == z.sizes(), "Input x and z must have the same shape.");

        torch::Tensor x_abs = torch::abs(x);
        torch::Tensor x_sign = torch::sign(x);

        torch::Tensor term1 = x_sign * torch::max(x_abs, torch::abs(z));
        torch::Tensor term2 = torch::sign(z) * torch::min(x_abs, torch::abs(z));

        return term1 + term2;
    }


    auto NailOr::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::nail_or(torch::zeros(10), torch::zeros(10));
    }
}
