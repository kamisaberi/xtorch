#include "include/transforms/general/lambda.h"

namespace xt::transforms::general {

    Lambda::Lambda() = default;


    Lambda::Lambda(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }


    auto Lambda::forward(std::initializer_list <torch::Tensor> tensors) -> std::any {
        std::vector <torch::Tensor> tensor_vec(tensors);
        torch::Tensor input = tensor_vec[0];

        torch::Tensor Lambda::forward(torch::Tensor input) const {
            return transform(input);
        }
    }


}