#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor hbm_loss(torch::Tensor x);
    class HBMLoss : xt::Module
    {
    public:
        HBMLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
