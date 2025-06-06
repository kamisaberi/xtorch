#pragma once

#include "../common.h"


namespace xt::transforms::graph
{
    class NodeMixUp final : public xt::Module
    {
    public:
        NodeMixUp();
        explicit NodeMixUp(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
