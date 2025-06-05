#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class Scale final : public xt::Module
    {
    public:
        Scale();
        explicit Scale(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
