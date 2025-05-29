#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class StyleMixing final : public xt::Module
    {
    public:
        StyleMixing();
        explicit StyleMixing(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
