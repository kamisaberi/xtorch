#pragma once



#include "../../headers/transforms.h"

namespace xt::transforms::image {


    struct Grayscale {
    public:
        Grayscale();
        torch::Tensor operator()(torch::Tensor input);
    };

    struct ToGray {
        torch::Tensor operator()(const torch::Tensor& color_tensor) const;
    };


}