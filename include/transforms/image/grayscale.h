//TODO SHOULD CHANGE
#pragma once


#include "../common.h"


namespace xt::transforms::image
{
    struct Grayscale
    {
    public:
        Grayscale();
        torch::Tensor operator()(torch::Tensor input);
    };
}
