#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor supervised_contrastive_loss(torch::Tensor x);
    class SupervisedContrastiveLoss : xt::Module
    {
    public:
        SupervisedContrastiveLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
