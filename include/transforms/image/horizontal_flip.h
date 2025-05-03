#pragma once
#include "../../headers/transforms.h"

namespace xt::transforms::image
{
    struct HorizontalFlip
    {
    public:
        HorizontalFlip();

        torch::Tensor operator()(torch::Tensor input);
    };
}
