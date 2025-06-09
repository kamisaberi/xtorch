#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor ghm_r(torch::Tensor x);
    class GHMR : xt::Module
    {
    public:
        GHMR() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
