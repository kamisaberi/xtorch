#include "include/losses/dsam_loss.h"

namespace xt::losses
{
    torch::Tensor dsam_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto DSAMLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::dsam_loss(torch::zeros(10));
    }
}
