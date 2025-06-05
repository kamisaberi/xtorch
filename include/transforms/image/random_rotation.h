#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomRotation final : public xt::Module
    {
    public:
        RandomRotation();
        explicit RandomRotation(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
