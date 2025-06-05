//TODO SHOULD CHANGE
#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    struct GaussianNoise
    {
    public:
        GaussianNoise(float mean, float std);

        torch::Tensor operator()(torch::Tensor input);

    private:
        float mean;
        float std;
    };
}
