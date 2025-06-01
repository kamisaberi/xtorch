#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::target
{
    class MinMaxScaler final : public xt::Module
    {
    public:
        MinMaxScaler();
        explicit MinMaxScaler(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
