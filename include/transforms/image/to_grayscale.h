#pragma once



#include "../../headers/transforms.h"

namespace xt::transforms::image {

    struct Grayscale {
        torch::Tensor operator()(const torch::Tensor& color_tensor) const;
    };


}