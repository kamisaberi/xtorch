#pragma once
#include "include/transforms/common.h"

namespace xt::transforms
{
    struct ToTensor final : xt::Module {
    public:
        ToTensor();
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        torch::Tensor operator()(const cv::Mat& image) const;
    };


}