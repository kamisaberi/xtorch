#pragma once

#include "../../headers/transforms.h"

namespace xt::transforms::image {


    struct VerticalFlip {
    public:
        VerticalFlip();

        torch::Tensor operator()(torch::Tensor input);
    };

}