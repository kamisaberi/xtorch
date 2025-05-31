#include "include/transforms/image/crop.h"

namespace xt::transforms::image {

    Crop::Crop() = default;

    Crop::Crop(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto Crop::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}