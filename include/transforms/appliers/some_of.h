#pragma once

#include "../common.h"

namespace xt::transforms
{
    class SomeOf : xt::Module
    {
    public:
        SomeOf();
        explicit SomeOf(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::vector<xt::Module> transforms;
    };
}
