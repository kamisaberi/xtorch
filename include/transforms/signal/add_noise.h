#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::signal
{
    class AddNoise final : public xt::Module
    {
    public:
        AddNoise();
        explicit AddNoise(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
