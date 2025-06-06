#pragma once

#include "../common.h"


namespace xt::transforms::target
{
    class TargetNormalizer final : public xt::Module
    {
    public:
        TargetNormalizer();
        explicit TargetNormalizer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
