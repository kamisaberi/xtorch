#include "include/transforms/image/crop.h"

namespace xt::transforms::image {

    Crop::Crop() = default;

    Crop::Crop(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto Crop::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}