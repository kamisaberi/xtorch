#include "include/transforms/image/random_resized_crop.h"

namespace xt::transforms::image
{
    RandomResizedCrop::RandomResizedCrop() = default;

    RandomResizedCrop::RandomResizedCrop(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomResizedCrop::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
