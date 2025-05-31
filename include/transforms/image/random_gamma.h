#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class RandomGamma final : public xt::Module
    {
    public:
        RandomGamma();
        explicit RandomGamma(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
