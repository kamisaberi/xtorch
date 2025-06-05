#pragma once

#include "../common.h"


namespace xt::transforms::graph
{
    class GraphMixUp final : public xt::Module
    {
    public:
        GraphMixUp();
        explicit GraphMixUp(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
