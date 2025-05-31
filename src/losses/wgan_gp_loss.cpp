#include "include/losses/wgan_gp_loss.h"

namespace xt::losses
{
    torch::Tensor wgan_gp_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto WGANGPLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::wgan_gp_loss(torch::zeros(10));
    }
}
