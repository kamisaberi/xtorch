#include "include/losses/nt_xent.h"

namespace xt::losses
{
    torch::Tensor nt_xent(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto NTXent::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::nt_xent(torch::zeros(10));
    }
}
