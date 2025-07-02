#pragma once
#include "../common.h"


namespace xt::transforms::general
{
    struct ToTensor final : xt::Module {
    public:
        ToTensor();
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        torch::Tensor operator()(const cv::Mat& image) const;
    };


}