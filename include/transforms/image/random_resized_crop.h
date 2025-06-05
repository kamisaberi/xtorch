#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomResizedCrop final : public xt::Module
    {
    public:
        RandomResizedCrop();
        explicit RandomResizedCrop(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
