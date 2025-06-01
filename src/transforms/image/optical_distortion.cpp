#include "include/transforms/image/optical_distortion.h"

namespace xt::transforms::image
{
    OpticalDistortion::OpticalDistortion() = default;

    OpticalDistortion::OpticalDistortion(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto OpticalDistortion::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
