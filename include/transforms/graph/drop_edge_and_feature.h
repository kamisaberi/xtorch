#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::graph
{
    class DropEdgeAndFeature final : public xt::Module
    {
    public:
        DropEdgeAndFeature();
        explicit DropEdgeAndFeature(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
