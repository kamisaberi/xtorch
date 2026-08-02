#pragma once
#include "common.h"


namespace xt::losses
{
    class UPITLoss : xt::Module
    {
    public:
        UPITLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
