#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomHorizontalFlip final : public xt::Module
    {
    public:
        RandomHorizontalFlip();
        explicit RandomHorizontalFlip(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
