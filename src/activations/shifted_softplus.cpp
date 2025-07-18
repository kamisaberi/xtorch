#include <activations/shifted_softplus.h>

namespace xt::activations
{

    torch::Tensor shifted_softplus(const torch::Tensor& x, double shift_val )
    {
        return torch::softplus(x) - shift_val;
        // torch::softplus(x) = torch::log1p(torch::exp(x))
    }

    auto ShiftedSoftplus::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::shifted_softplus(torch::zeros(10));
    }
}
