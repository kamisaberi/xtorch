#include "include/transforms/video/optical_flow_warping.h"

namespace xt::transforms::video
{
    OpticalFlowWarping::OpticalFlowWarping() = default;

    OpticalFlowWarping::OpticalFlowWarping(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto OpticalFlowWarping::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
