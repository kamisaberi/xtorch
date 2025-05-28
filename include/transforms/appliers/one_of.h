#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class OneOf final : public xt::Module
    {
    public:
        OneOf();
        explicit OneOf(std::vector<xt::Module> transforms);
//        torch::Tensor forward(torch::Tensor input) const override;
        auto forward(std::initializer_list<torch::Tensor> tensors) -> std::any  override;

    private:
        std::vector<xt::Module> transforms;

    };
}
