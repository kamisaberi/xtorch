#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class NodeFeatureMasking final : public xt::Module
    {
    public:
        NodeFeatureMasking();
        explicit NodeFeatureMasking(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
