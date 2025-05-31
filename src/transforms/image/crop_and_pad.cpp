#include "include/transforms/image/crop_and_pad.h"

namespace xt::transforms::image {

    CropAndPad::CropAndPad() = default;

    CropAndPad::CropAndPad(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto CropAndPad::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}