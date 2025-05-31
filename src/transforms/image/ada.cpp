#include "include/transforms/image/ada.h"

namespace xt::transforms::image {

    ADA::ADA() = default;

    ADA::ADA(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto ADA::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}