#include "include/transforms/general/pca.h"

namespace xt::transforms::general {
    PCA::PCA() = default;

    PCA::PCA(torch::ScalarType target_dtype) : xt::Module() {
    }


    auto PCA::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }

}
