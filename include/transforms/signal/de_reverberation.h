#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class DeReverberation final : public xt::Module
    {
    public:
        DeReverberation();
        explicit EdgeAdd(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
