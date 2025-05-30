#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor proxy_anchor_loss(torch::Tensor x);
    class ProxyAnchorLoss : xt::Module
    {
    public:
        ProxyAnchorLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
