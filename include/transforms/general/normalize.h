#pragma once

#include "transforms/common.h"

namespace xt::transforms
{
    struct Normalize
    {
    public:
        Normalize(std::vector<float> mean, std::vector<float> std);
        torch::Tensor operator()(const torch::Tensor& tensor) const;

    private:
        std::vector<float> mean;
        std::vector<float> std;
    };
}
