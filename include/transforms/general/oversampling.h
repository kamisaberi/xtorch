#pragma once
#include "include/transforms/common.h"



namespace xt::transforms::general {

    struct OverSampling final : xt::Module {
    public:
        OverSampling();
        OverSampling(std::function<torch::Tensor(torch::Tensor)> transform);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}
