#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor early_exiting(torch::Tensor x);
    class EarlyExiting : xt::Module
    {
    public:
        EarlyExiting() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
