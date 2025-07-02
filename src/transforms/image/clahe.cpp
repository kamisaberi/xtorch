#include "include/transforms/image/clahe.h"

namespace xt::transforms::image {

    CLAHE::CLAHE() = default;

    CLAHE::CLAHE(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto CLAHE::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}