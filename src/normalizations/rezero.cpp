#include "include/normalizations/rezero.h"

namespace xt::norm
{
    auto Rezero::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
