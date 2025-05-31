#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class ToSepia final : public xt::Module
    {
    public:
        ToSepia();
        explicit ToSepia(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
