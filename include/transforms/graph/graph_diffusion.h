#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::graph
{
    class GraphDiffusion final : public xt::Module
    {
    public:
        GraphDiffusion();
        explicit GraphDiffusion(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
