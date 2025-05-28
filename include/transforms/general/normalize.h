#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::general {


    struct Normalize final : xt::Module {
    public:
        Normalize();

//        Normalize(float mean, float std);

        Normalize(std::vector<float> mean, std::vector<float> std);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::vector<float> mean;
        std::vector<float> std;
    };
}
