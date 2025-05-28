#pragma once
#include "include/transforms/common.h"

namespace xt::transforms::general {

    struct RandomOverSampling final : xt::Module {
    public:
        RandomOverSampling();
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}