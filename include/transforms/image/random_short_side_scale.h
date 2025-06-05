#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomShortSideScale final : public xt::Module
    {
    public:
        RandomShortSideScale();
        explicit RandomShortSideScale(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
