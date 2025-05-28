
#include "include/transforms/general/convert_dtype.h"

namespace xt::transforms::general {
    ConvertDType::ConvertDType() = default;

    ConvertDType::ConvertDType(torch::ScalarType target_dtype) : xt::Module(), target_dtype(target_dtype) {
    }


    auto ConvertDTypeCompose::forward(std::initializer_list <torch::Tensor> tensors) -> std::any {

        std::vector <torch::Tensor> tensor_vec(tensors);
        torch::Tensor input = tensor_vec[0];
        if (!input.defined()) {
            throw std::invalid_argument("Input tensor is not defined");
        }

        if (input.dtype() == target_dtype) {
            return input;
        }

        return input.to(target_dtype);

    }

}
