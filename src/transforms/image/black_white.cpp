#include "include/transforms/image/black_white.h"

namespace xt::transforms::image  {

    BlackWhite::BlackWhite() = default;

    BlackWhite::BlackWhite(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto BlackWhite::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}