#include "include/transforms/image/random_thin_plate_spline.h"

namespace xt::transforms::image
{
    RandomThinPlateSpline::RandomThinPlateSpline() = default;

    RandomThinPlateSpline::RandomThinPlateSpline(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomThinPlateSpline::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
