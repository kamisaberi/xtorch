#pragma once

#include "../common.h"


namespace xt::transforms::graph
{
    class EdgeFeatureMasking final : public xt::Module
    {
    public:
        EdgeFeatureMasking();
        explicit EdgeFeatureMasking(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
