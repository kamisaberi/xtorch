#include "include/transforms/image/coarse_dropout.h"

namespace xt::transforms::image {

    CoarseDropout::CoarseDropout() = default;

    CoarseDropout::CoarseDropout(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto CoarseDropout::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}