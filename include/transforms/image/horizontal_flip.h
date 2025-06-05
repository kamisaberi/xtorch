//TODO SHOULD CHANGE
#pragma once
#include "../common.h"


namespace xt::transforms::image
{
    struct HorizontalFlip
    {
    public:
        HorizontalFlip();

        torch::Tensor operator()(torch::Tensor input);
    };
}
