#pragma once

#include "../../headers/transforms.h"

namespace xt::data::transforms {
    struct Normalize {
    public:
        Normalize(std::vector<float> mean, std::vector<float> std);
        torch::Tensor operator()(const torch::Tensor& tensor) const;
    private:
        std::vector<float> mean;
        std::vector<float> std;

    };

}