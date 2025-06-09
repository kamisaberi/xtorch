#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor happier(torch::Tensor x);
    class HAPPIER : xt::Module
    {
    public:
        HAPPIER() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
