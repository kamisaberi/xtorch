#pragma once
#include "../common.h"




namespace xt::transforms::general {

    struct OverSampling final : xt::Module {
    public:
        OverSampling();
        OverSampling(std::function<torch::Tensor(torch::Tensor)> transform);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}
