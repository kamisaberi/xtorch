#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class StandardScaler final : public xt::Module
    {
    public:
        StandardScaler();
        explicit StandardScaler(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
