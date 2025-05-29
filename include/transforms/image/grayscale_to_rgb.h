//TODO SHOULD CHANGE
#pragma once


#include "include/transforms/common.h"

namespace xt::transforms::image {

    struct GrayscaleToRGB {
    public:
        torch::Tensor operator()(const torch::Tensor &tensor);
    };


}