#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class GridDropout final : public xt::Module
    {
    public:
        GridDropout();
        explicit GridDropout(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
