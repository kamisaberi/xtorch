#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomGrayscale final : public xt::Module
    {
    public:
        RandomGrayscale();
        explicit RandomGrayscale(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
