#pragma once
#include "include/transforms/common.h"

namespace xt::transforms::general {

    struct RandomOverSampling final : xt::Module {
    public:
        RandomOverSampling();
        RandomOverSampling(std::function<torch::Tensor(torch::Tensor)> transform);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}