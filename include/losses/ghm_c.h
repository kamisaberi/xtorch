#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor ghmc(torch::Tensor x);
    class GHMC : xt::Module
    {
    public:
        GHMC() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
