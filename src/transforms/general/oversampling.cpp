#include "include/transforms/general/oversampling.h"

namespace xt::transforms::general {

    OverSampling::OverSampling() = default;


    OverSampling::OverSampling(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto OverSampling::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}