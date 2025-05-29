#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Equalize final : public xt::Module
    {
    public:
        Equalize();
        explicit Equalize(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
