#pragma once
#include "transforms/common.h"

namespace xt::transforms::general {

    struct Lambda final : xt::Module {
    public:
        Lambda();
        Lambda(std::function<torch::Tensor(torch::Tensor)> transform);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}