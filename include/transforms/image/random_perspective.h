#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomPerspective final : public xt::Module
    {
    public:
        RandomPerspective();
        explicit RandomPerspective(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
