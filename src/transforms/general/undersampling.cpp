#include "transforms/general/undersampling.h"

namespace xt::transforms::general {

    UnderSampling::UnderSampling() = default;


    UnderSampling::UnderSampling(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    torch::Tensor Lambda::forward(torch::Tensor input) {
        return transform(input);
    }

}