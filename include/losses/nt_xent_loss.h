#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor nt_xent_loss(torch::Tensor x);
    class NTXentLoss : xt::Module
    {
    public:
        NTXentLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
