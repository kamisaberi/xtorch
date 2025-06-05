#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomThinPlateSpline final : public xt::Module
    {
    public:
        RandomThinPlateSpline();
        explicit RandomThinPlateSpline(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
