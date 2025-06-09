#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor lovasz_softmax_loss(torch::Tensor x);
    class LovaszSoftmaxLoss : xt::Module
    {
    public:
        LovaszSoftmaxLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
