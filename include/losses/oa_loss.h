#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor object_aware_loss(torch::Tensor x);
    class ObjectAwareLoss : xt::Module
    {
    public:
        ObjectAwareLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
