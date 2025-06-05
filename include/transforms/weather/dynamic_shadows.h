#pragma once

#include "../common.h"


namespace xt::transforms::weather
{
    class DynamicShadows final : public xt::Module
    {
    public:
        DynamicShadows();
        explicit DynamicShadows(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
