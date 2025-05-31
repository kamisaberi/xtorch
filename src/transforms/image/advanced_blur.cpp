#include "include/transforms/image/advanced_blur.h"

namespace xt::transforms::image  {

    AdvancedBlur::AdvancedBlur() = default;

    AdvancedBlur::AdvancedBlur(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto AdvancedBlur::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}