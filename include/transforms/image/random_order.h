#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomOrder final : public xt::Module
    {
    public:
        RandomOrder();
        explicit RandomOrder(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
