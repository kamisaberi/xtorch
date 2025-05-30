#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor generalized_focal_loss(torch::Tensor x);
    class GeneralizedFocalLoss : xt::Module
    {
    public:
        GeneralizedFocalLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
