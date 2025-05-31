#include "include/transforms/image/brightness.h"

namespace xt::transforms::image {

    Brightness::Brightness() = default;

    Brightness::Brightness(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto Brightness::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}