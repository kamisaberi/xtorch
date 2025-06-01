#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::target
{
    class BinaryEncoder final : public xt::Module
    {
    public:
        BinaryEncoder();
        explicit BinaryEncoder(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
