#include "include/losses/gan_least_squares_loss.h"

namespace xt::losses
{
    torch::Tensor gan_least_squares_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto GANLeastSquaresLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::gan_least_squares_loss(torch::zeros(10));
    }
}
