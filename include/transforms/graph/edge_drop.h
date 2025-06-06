#pragma once

#include "../common.h"


namespace xt::transforms::graph
{
    class EdgeDrop final : public xt::Module
    {
    public:
        EdgeDrop();
        explicit EdgeDrop(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
