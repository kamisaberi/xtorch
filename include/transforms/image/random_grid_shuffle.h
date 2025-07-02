#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomGridShuffle final : public xt::Module
    {
    public:
        RandomGridShuffle();
        explicit RandomGridShuffle(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
