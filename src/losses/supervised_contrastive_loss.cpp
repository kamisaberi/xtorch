#include "include/losses/supervised_contrastive_loss.h"

namespace xt::losses
{
    torch::Tensor supervised_contrastive_loss(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SupervisedContrastiveLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::supervised_contrastive_loss(torch::zeros(10));
    }
}
