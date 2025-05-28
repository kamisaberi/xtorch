#pragma once
#include "include/transforms/common.h"

namespace xt::transforms
{
    struct ToTensor {
    public:
        auto forward(std::initializer_list<torch::Tensor> tensors) -> std::any  override;
        torch::Tensor operator()(const cv::Mat& image) const;
    };


}