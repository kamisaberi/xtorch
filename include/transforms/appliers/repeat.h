#pragma once

#include "../common.h"

namespace xt::transforms
{
    class Repeat : public xt::Module
    {
    public:
        Repeat();
        explicit Repeat(std::unique_ptr<xt::Module> transform, int n_times);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
    private:
        std::unique_ptr<xt::Module> transform;
        int n_times_;
    };
}
