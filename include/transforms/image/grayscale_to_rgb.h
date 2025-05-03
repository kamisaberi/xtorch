#pragma once



#include "../headers/transforms.h"

namespace xt::transforms::image {

    struct GrayscaleToRGB {
    public:
        torch::Tensor operator()(const torch::Tensor &tensor);
    };


}