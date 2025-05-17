#include "transforms/general/pca.h"

namespace xt::transforms::general {

    PCA::PCA() = default;

    PCA::PCA(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    torch::Tensor Lambda::forward(torch::Tensor input) {
        return transform(input);
    }

}