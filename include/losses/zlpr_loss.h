#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor zlpr_loss(torch::Tensor x);
    class ZLPRLoss : xt::Module
    {
    public:
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
