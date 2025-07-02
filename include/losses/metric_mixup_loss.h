#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor metric_mixup_loss(torch::Tensor x);

    class MetricMixupLoss : xt::Module
    {
    public:
        MetricMixupLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
