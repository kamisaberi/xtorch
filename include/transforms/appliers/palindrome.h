#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Palindrome final :public xt::Module
    {
    public:
        Palindrome();
        explicit Palindrome(std::vector<xt::Module> transforms);
//        torch::Tensor forward(torch::Tensor input) const override;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
        std::vector<xt::Module> transforms;
    };
}
