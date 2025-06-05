//TODO SHOULD CHANGE
#pragma once
#include "../common.h"


namespace xt::transforms::image
{
    struct VerticalFlip
    {
    public:
        VerticalFlip();

        torch::Tensor operator()(torch::Tensor input);
    };
}
