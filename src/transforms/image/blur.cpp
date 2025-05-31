#include "include/transforms/image/blur.h"

namespace xt::transforms::image {

    Blur::Blur() = default;

    Blur::Blur(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto Blur::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}