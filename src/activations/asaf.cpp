#include "include/activations/asaf.h"

namespace xt::activations
{
    torch::Tensor asaf(const torch::Tensor x, double p_param, double q_param)
    {
        torch::Tensor sig_px = torch::sigmoid(p_param * x);
        torch::Tensor sig_qx = torch::sigmoid(q_param * x);
        return x * sig_px + sig_qx - x;
    }

    auto ASAF::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::asaf(torch::zeros(10));
    }
}
