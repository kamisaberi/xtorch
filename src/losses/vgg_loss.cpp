#include "include/losses/vgg_loss.h"

namespace xt::losses
{
    torch::Tensor vgg_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto VGGLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::vgg_loss(torch::zeros(10));
    }
}
