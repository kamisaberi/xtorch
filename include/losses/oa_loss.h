#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor oa_loss(torch::Tensor x);
    class OALoss : xt::Module
    {
    public:
        OALoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
