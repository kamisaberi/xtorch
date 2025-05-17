#pragma once

#include "transforms/common.h"

namespace xt::transforms::general {


    struct Normalize::xt::Module {
    public:
        Normalize();

//        Normalize(float mean, float std);

        Normalize(std::vector<float>  mean, std::vector<float> std);

        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::vector<float> mean;
        std::vector<float> std;
    };
}
