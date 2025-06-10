#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor unsupervised_feature_loss(torch::Tensor x);
    class UnsupervisedFeatureLoss : xt::Module
    {
    public:
        UnsupervisedFeatureLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
