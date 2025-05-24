#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Palindrome final :public xt::Module
    {
    public:
        Palindrome();
        explicit Palindrome(std::vector<xt::Module> transforms);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::vector<xt::Module> transforms;
    };
}
