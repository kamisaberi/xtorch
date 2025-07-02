#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class Solarize final : public xt::Module
    {
    public:
        Solarize();
        explicit Solarize(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
