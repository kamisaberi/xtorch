#include "include/transforms/text/back_translation.h"

namespace xt::transforms::text
{
    BackTranslation::BackTranslation() = default;

    BackTranslation::BackTranslation(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto BackTranslation::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
