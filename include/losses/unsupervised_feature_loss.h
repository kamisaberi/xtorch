#pragma once
#include "common.h"


namespace xt::losses
{
    class UnsupervisedFeatureLoss : xt::Module
    {
    public:
        UnsupervisedFeatureLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
