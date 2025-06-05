#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomSolarize final : public xt::Module
    {
    public:
        RandomSolarize();
        explicit RandomSolarize(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
