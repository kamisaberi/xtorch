#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor flip(torch::Tensor x);
    class FLIP : xt::Module
    {
    public:

        FLIP() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
