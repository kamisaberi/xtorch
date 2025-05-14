#pragma once

#include "transforms/common.h"

namespace xt::transforms
{
    class RandomApply final : public xt::Module
    {
    public:
        using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;
        RandomApply();
        explicit RandomApply(std::vector<TransformFunc> transforms);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::vector<TransformFunc> transforms;
    };
}
