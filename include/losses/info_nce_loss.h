#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor info_nce(torch::Tensor x);
    class InfoNCE : xt::Module
    {
    public:
        InfoNCE() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
