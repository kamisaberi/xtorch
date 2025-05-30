#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor metrix(torch::Tensor x);
    class Metrix : xt::Module
    {
    public:
        Metrix() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
