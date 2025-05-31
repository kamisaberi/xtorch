#include "include/transforms/graph/node_mix_up.h"

namespace xt::transforms::graph {

    NodeMixUp::NodeMixUp() = default;

    NodeMixUp::NodeMixUp(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto NodeMixUp::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}