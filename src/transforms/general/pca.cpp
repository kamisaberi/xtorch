#include "include/transforms/general/pca.h"

namespace xt::transforms::general {
    PCA::PCA() = default;

    PCA::PCA(std::vector<xt::Module> transforms) : xt::Module() {
    }


    auto PCA::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }

}
