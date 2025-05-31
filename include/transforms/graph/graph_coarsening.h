#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::graph
{
    class GraphCoarsening final : public xt::Module
    {
    public:
        GraphCoarsening();
        explicit GraphCoarsening(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
