#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomEqualize final : public xt::Module
    {
    public:
        RandomEqualize();
        explicit RandomEqualize(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
