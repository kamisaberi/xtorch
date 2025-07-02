#include "include/transforms/general/undersampling.h"

namespace xt::transforms::general {

    UnderSampling::UnderSampling() = default;

    auto UnderSampling::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


    // UnderSampling::UnderSampling(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    // }
    //
    // torch::Tensor UnderSampling::forward(torch::Tensor input) const {
    //     return transform(input);
    // }

}