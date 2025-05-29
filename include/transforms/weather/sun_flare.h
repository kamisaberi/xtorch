#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class SunFlare final : public xt::Module
    {
    public:
        SunFlare();
        explicit SunFlare(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
