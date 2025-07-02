#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class AdvancedBlur final : public xt::Module
    {
    public:
        AdvancedBlur();
        explicit AdvancedBlur(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
