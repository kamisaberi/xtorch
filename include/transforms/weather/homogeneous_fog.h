#pragma once

#include "../common.h"


namespace xt::transforms::weather
{
    class HomogeneousFog final : public xt::Module
    {
    public:
        HomogeneousFog();
        explicit HomogeneousFog(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
