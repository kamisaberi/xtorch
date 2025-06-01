#include "include/transforms/image/random_crop_near_bbox.h"

namespace xt::transforms::image
{
    RandomCropNearBbox::RandomCropNearBbox() = default;

    RandomCropNearBbox::RandomCropNearBbox(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomCropNearBbox::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
