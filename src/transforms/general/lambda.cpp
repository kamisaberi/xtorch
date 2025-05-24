#include "include/transforms/general/lambda.h"

namespace xt::transforms::general {

    Lambda::Lambda() = default;


    Lambda::Lambda(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    torch::Tensor Lambda::forward(torch::Tensor input) const{
        return transform(input);
    }

}