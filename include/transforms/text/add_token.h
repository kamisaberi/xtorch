#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class AddToken final : public xt::Module
    {
    public:
        AddToken();
        explicit AddToken(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
