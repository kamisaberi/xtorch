#include "include/transforms/graph/drop_edge_and_feature.h"

namespace xt::transforms::graph {

    DropEdgeAndFeature::DropEdgeAndFeature() = default;


    DropEdgeAndFeature::DropEdgeAndFeature(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto DropEdgeAndFeature::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}