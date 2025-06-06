#pragma once

#include "../common.h"


namespace xt::transforms::graph
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
