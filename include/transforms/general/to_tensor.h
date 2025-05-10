#pragma once
#include "../../headers/transforms.h"

namespace xt::transforms
{
    struct ToTensor {
    public:
        torch::Tensor operator()(const cv::Mat& image) const;
    };


}