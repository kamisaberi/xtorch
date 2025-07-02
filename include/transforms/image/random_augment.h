#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomAugment final : public xt::Module
    {
    public:
        RandomAugment();
        explicit RandomAugment(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
