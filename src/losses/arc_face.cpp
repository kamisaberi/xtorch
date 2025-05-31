#include "include/losses/arc_face.h"

namespace xt::losses
{
    torch::Tensor arc_Face(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ArcFace::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::arc_Face(torch::zeros(10));
    }
}
