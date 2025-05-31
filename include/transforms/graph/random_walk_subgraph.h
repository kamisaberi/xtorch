#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::graph
{
    class RandomWalkSubgraph final : public xt::Module
    {
    public:
        RandomWalkSubgraph();
        explicit RandomWalkSubgraph(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
