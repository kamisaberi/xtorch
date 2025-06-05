#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class GridShuffle final : public xt::Module
    {
    public:
        GridShuffle();
        explicit GridShuffle(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
