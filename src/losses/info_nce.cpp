#include "include/losses/info_nce.h"

namespace xt::losses
{
    torch::Tensor info_nce(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto InfoNCE::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::info_nce(torch::zeros(10));
    }
}
