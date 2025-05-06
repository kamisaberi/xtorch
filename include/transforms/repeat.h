#pragma once

#include "../headers/transforms.h"

namespace xt::transforms
{
    class Repeat : public xt::Module
    {
    public:
        using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;
        Repeat();
        explicit Repeat(TransformFunc transform, int n_times);
        torch::Tensor operator()(torch::Tensor input)  const override;

    private:
        TransformFunc transform;
        int n_times_;
    };
}
