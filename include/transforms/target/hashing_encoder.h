#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::target
{
    class HashingEncoder final : public xt::Module
    {
    public:
        HashingEncoder();
        explicit HashingEncoder(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
