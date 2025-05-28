#pragma once
#include "include/transforms/common.h"

namespace xt::transforms::general {

    struct RandomUnderSampling final : xt::Module {
    public:
        RandomUnderSampling();
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}