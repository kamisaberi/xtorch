//TODO SHOULD CHANGE
#pragma once
#include "include/transforms/common.h"

namespace xt::transforms::image
{
    struct HorizontalFlip
    {
    public:
        HorizontalFlip();

        torch::Tensor operator()(torch::Tensor input);
    };
}
