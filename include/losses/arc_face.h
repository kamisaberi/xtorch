#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor arc_Face(torch::Tensor x);

    class ArcFace : xt::Module
    {
    public:
        ArcFace() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
