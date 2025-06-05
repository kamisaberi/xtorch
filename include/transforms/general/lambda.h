#pragma once
#include "../common.h"


namespace xt::transforms::general {

    struct Lambda final : xt::Module {
    public:
        Lambda();
        Lambda(std::function<torch::Tensor(torch::Tensor)> transform);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}