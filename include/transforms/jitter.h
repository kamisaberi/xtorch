#pragma once

#include "../headers/transforms.h"

namespace xt::data::transforms {


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