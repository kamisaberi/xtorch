#include "include/transforms/text/str_to_int_transform.h"

namespace xt::transforms::text
{
    StrToIntTransform::StrToIntTransform() = default;

    StrToIntTransform::StrToIntTransform(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto StrToIntTransform::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
