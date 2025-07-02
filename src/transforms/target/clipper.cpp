#include "include/transforms/target/clipper.h"

namespace xt::transforms::target
{
    Clipper::Clipper() = default;

    Clipper::Clipper(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Clipper::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
