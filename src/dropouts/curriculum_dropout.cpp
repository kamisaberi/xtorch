#include "include/dropouts/curriculum_dropout.h"

namespace xt::dropouts
{
    torch::Tensor curriculum_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto CurriculumDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::curriculum_dropout(torch::zeros(10));
    }
}
