#include "include/transforms/graph/graph_diffusion.h"

namespace xt::transforms::graph
{
    GraphDiffusion::GraphDiffusion() = default;

    GraphDiffusion::GraphDiffusion(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto GraphDiffusion::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
