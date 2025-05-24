#pragma once
#include "include/transforms/common.h"



namespace xt::transforms::general {

    struct UnderSampling final : xt::Module {
    public:
        UnderSampling();
        UnderSampling(std::function<torch::Tensor(torch::Tensor)> transform);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}
