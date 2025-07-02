#pragma once

#include "../common.h"


namespace xt::transforms::text
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
