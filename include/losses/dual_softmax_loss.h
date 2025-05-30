#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor dual_softmax_loss(torch::Tensor x);
    class DualSoftmaxLoss : xt::Module
    {
    public:
        DualSoftmaxLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
