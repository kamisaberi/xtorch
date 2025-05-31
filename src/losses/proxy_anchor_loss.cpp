#include "include/losses/proxy_anchor_loss.h"

namespace xt::losses
{
    torch::Tensor proxy_anchor_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ProxyAnchorLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::proxy_anchor_loss(torch::zeros(10));
    }
}
