#include <activations/ahaf.h>

namespace xt::activations
{
    torch::Tensor ahaf(const torch::Tensor x, double p_param)
    {
        torch::Tensor p_x = p_param * x;
        torch::Tensor tanh_p_x = torch::tanh(p_x);
        return x * (1.0 + tanh_p_x) / 2.0;
    }

    auto AHAF::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::ahaf(torch::zeros(10));
    }
}
