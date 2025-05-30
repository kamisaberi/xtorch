#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor uf_loss(torch::Tensor x);
    class UFLoss : xt::Module
    {
    public:
        UFLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
