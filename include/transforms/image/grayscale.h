#pragma once



#include "../../headers/transforms.h"

namespace xt::transforms::image {

    struct Grayscale {
    public:
        Grayscale();
        torch::Tensor operator()(torch::Tensor input);
    };

}