#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::signal
{
    class MelScale final : public xt::Module
    {
    public:
        MelScale();
        explicit MelScale(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
