#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class JamesSteinEncoder final : public xt::Module
    {
    public:
        JamesSteinEncoder();
        explicit JamesSteinEncoder(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
