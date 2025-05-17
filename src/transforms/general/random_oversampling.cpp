#include "transforms/general/random_oversampling.h"

namespace xt::transforms::general {

    RandomOverSampling::RandomOverSampling() = default;


    RandomOverSampling::RandomOverSampling(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    torch::Tensor Lambda::forward(torch::Tensor input) {
        return transform(input);
    }

}