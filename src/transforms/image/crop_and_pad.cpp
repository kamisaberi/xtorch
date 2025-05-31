#include "include/transforms/image/crop_and_pad.h"

namespace xt::transforms::image {

    CropAndPad::CropAndPad() = default;

    CropAndPad::CropAndPad(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto CropAndPad::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}