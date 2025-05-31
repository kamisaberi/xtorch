#include "include/losses/upit.h"

namespace xt::losses
{
    torch::Tensor upit(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto UPIT::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::upit(torch::zeros(10));
    }
}
