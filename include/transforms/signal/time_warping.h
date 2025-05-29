#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class TimeWarping final : public xt::Module
    {
    public:
        TimeWarping();
        explicit TimeWarping(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
