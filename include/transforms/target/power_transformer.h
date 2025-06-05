#pragma once

#include "../common.h"


namespace xt::transforms::target
{
    class PowerTransformer final : public xt::Module
    {
    public:
        PowerTransformer();
        explicit PowerTransformer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
