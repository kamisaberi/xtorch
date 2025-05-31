#include "include/transforms/image/clahe.h"

namespace xt::transforms::image {

    CLAHE::CLAHE() = default;

    CLAHE::CLAHE(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto CLAHE::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}