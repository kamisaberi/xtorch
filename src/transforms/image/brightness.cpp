#include "include/transforms/image/brightness.h"

namespace xt::transforms::image {

    Brightness::Brightness() = default;

    Brightness::Brightness(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto Brightness::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}