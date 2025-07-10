#pragma once

#include "module.h"
#include <vector>
#include <algorithm>
#include <iostream>

namespace xt
{
    class Sequential : xt::Module
    {
    public:
        Sequential();
        Sequential(std::initializer_list<xt::Module> blocks);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        torch::Tensor forward(torch::Tensor x);

    private:
        std::vector<xt::Module> blocks_;
    };
}
