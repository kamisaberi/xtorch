#include "include/transforms/general/oversampling.h"

namespace xt::transforms::general {

    OverSampling::OverSampling() = default;


    OverSampling::OverSampling(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    // torch::Tensor OverSampling::forward(torch::Tensor input) const {
    //     return transform(input);
    // }

}