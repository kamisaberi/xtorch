#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class NodeDrop final : public xt::Module
    {
    public:
        NodeDrop();
        explicit NodeDrop(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
