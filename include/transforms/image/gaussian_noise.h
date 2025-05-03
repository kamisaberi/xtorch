#pragma once
#include "../../headers/transforms.h"

namespace xt::data::transforms {

    struct GaussianNoise {
    public:
        GaussianNoise(float mean, float std);

        torch::Tensor operator()(torch::Tensor input);

    private:
        float mean;
        float std;
    };


}