#pragma once
#include "include/transforms/common.h"

namespace xt::transforms::general {

    struct RandomOverSampling final : xt::Module {
    public:
        RandomOverSampling();
        RandomOverSampling(std::function<torch::Tensor(torch::Tensor)> transform);
        auto forward(std::initializer_list<torch::Tensor> tensors) -> std::any  override;

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}