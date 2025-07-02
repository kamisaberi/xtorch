#pragma once

#include "../common.h"


namespace xt::transforms::graph
{
    class NodeFeatureShuffling final : public xt::Module
    {
    public:
        NodeFeatureShuffling();
        explicit NodeFeatureShuffling(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
