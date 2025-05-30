#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor seesaw_loss(torch::Tensor x);
    class SeesawLoss : xt::Module
    {
    public:
        SeesawLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
