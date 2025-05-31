#include "include/losses/early_exiting.h"

namespace xt::losses
{
    torch::Tensor early_exiting(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto EarlyExiting::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::early_exiting(torch::zeros(10));
    }
}
