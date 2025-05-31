#include "include/transforms/image/crop_non_empty_mask_if_exists.h"

namespace xt::transforms::image
{
    CropNonEmptyMaskIfExists::CropNonEmptyMaskIfExists() = default;

    CropNonEmptyMaskIfExists::CropNonEmptyMaskIfExists(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto CropNonEmptyMaskIfExists::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
