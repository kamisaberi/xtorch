#include "include/transforms/graph/graph_mix_up.h"

namespace xt::transforms::graph {

    GraphMixUp::GraphMixUp() = default;

    GraphMixUp::GraphMixUp(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto GraphMixUp::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}