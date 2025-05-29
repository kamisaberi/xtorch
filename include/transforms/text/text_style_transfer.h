#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class TextStyleTransfer final : public xt::Module
    {
    public:
        TextStyleTransfer();
        explicit TextStyleTransfer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
