#pragma once

#include "../common.h"

namespace xt::transforms
{
    class RandomApply final : public xt::Module
    {
    public:
        RandomApply();
        explicit RandomApply(std::vector<xt::Module> transforms);
//        torch::Tensor forward(torch::Tensor input) const override;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::vector<xt::Module> transforms;
    };
}
