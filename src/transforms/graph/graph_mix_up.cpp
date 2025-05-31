#include "include/transforms/graph/graph_mix_up.h"

namespace xt::transforms::graph {

    GraphMixUp::GraphMixUp() = default;

    GraphMixUp::GraphMixUp(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto GraphMixUp::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}