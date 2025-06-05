#pragma once

#include "../common.h"


namespace xt::transforms::graph
{
    class EdgeAdd final : public xt::Module
    {
    public:
        EdgeAdd();
        explicit EdgeAdd(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
