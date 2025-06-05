//TODO SHOULD CHANGE
#pragma once


#include "../common.h"


namespace xt::transforms::image
{
    struct GrayscaleToRGB
    {
    public:
        torch::Tensor operator()(const torch::Tensor& tensor);
    };
}
