#include "include/losses/ghm_r.h"

namespace xt::losses
{
    torch::Tensor ghm_r(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto GHMR::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::ghm_r(torch::zeros(10));
    }
}
