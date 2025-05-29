//TODO SHOULD CHANGE
#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image {


    struct ColorJitter {
    public:
        ColorJitter(float brightness = 0.0f,
                    float contrast = 0.0f,
                    float saturation = 0.0f);

        torch::Tensor operator()(const torch::Tensor& input_tensor) const;
    private:

        float brightness;
        float contrast;
        float saturation;

    };

}