#pragma once

#include "../headers/transforms.h"

namespace xt::transforms
{
    class OneOf : xt::Module
    {
    public:
        using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;
        OneOf();
        explicit OneOf(std::vector<TransformFunc> transforms);
        torch::Tensor operator()(torch::Tensor input) const override;

    private:
        std::vector<TransformFunc> transforms;
    };
}
