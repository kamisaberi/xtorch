#include "../../../include/transforms/general/lambda.h"

namespace xt::transforms {


    Lambda::Lambda(std::function<torch::Tensor(torch::Tensor)> transform)
        : transform(transform) {
    }

    torch::Tensor Lambda::operator()(torch::Tensor input) {
        return transform(input);
    }



}