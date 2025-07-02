#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomRotate45 final : public xt::Module
    {
    public:
        RandomRotate45();
        explicit RandomRotate45(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
