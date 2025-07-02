#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class CoarseDropout final : public xt::Module
    {
    public:
        CoarseDropout();
        explicit CoarseDropout(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
