#include "include/transforms/graph/graph_difussion.h"

namespace xt::transforms::graph {

    GraphDiffusion::GraphDiffusion() = default;

    GraphDiffusion::GraphDiffusion(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto GraphDiffusion::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}