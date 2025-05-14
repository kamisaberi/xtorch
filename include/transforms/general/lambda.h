#pragma once
#include "transforms/common.h"

namespace xt::transforms {

    struct Lambda {
    public:
        Lambda(std::function<torch::Tensor(torch::Tensor)> transform);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}