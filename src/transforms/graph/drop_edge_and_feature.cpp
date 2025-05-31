#include "include/transforms/graph/drop_edge_and_feature.h"

namespace xt::transforms::graph {

    DropEdgeAndFeature::DropEdgeAndFeature() = default;


    DropEdgeAndFeature::DropEdgeAndFeature(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto DropEdgeAndFeature::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}