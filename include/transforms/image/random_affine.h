#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomAffine final : public xt::Module
    {
    public:
        RandomAffine();
        explicit RandomAffine(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
