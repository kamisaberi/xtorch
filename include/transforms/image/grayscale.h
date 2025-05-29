//TODO SHOULD CHANGE
#pragma once



#include "include/transforms/common.h"

namespace xt::transforms::image {

    struct Grayscale {
    public:
        Grayscale();
        torch::Tensor operator()(torch::Tensor input);
    };

}