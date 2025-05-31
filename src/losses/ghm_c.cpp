#include "include/losses/ghm_c.h"

namespace xt::losses
{
    torch::Tensor ghmc(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto GHMC::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::ghmc(torch::zeros(10));
    }
}
