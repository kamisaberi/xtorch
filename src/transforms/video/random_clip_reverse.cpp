#include "include/transforms/video/random_clip_reverse.h"

namespace xt::transforms::video
{
    RandomClipReverse::RandomClipReverse() = default;

    RandomClipReverse::RandomClipReverse(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomClipReverse::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
