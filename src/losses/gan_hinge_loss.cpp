#include "include/losses/gan_hinge_loss.h"

namespace xt::losses
{
    torch::Tensor gan_hinge_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto GANHingeLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::gan_hinge_loss(torch::zeros(10));
    }
}
