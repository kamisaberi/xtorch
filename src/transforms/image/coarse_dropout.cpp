#include "include/transforms/image/coarse_dropout.h"

namespace xt::transforms::image {

    CoarseDropout::CoarseDropout() = default;

    CoarseDropout::CoarseDropout(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto CoarseDropout::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}