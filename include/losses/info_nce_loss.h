#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor info_nce_loss(torch::Tensor x);
    class InfoNCELoss : xt::Module
    {
    public:
        InfoNCELoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
