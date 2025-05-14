#pragma once
#include "transforms/common.h"

namespace xt::transforms
{
    struct ToTensor {
    public:
        torch::Tensor operator()(const cv::Mat& image) const;
    };


}