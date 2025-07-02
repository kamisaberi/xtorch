#include "include/transforms/graph/node_mix_up.h"

namespace xt::transforms::graph {

    NodeMixUp::NodeMixUp() = default;

    NodeMixUp::NodeMixUp(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto NodeMixUp::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}