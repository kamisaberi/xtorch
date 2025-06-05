#pragma once

#include "../common.h"


namespace xt::transforms::target
{
    class LabelBalancer final : public xt::Module
    {
    public:
        LabelBalancer();
        explicit LabelBalancer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
