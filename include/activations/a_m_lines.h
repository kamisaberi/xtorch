#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor am_lines(torch::Tensor x);

    struct AMLines : xt::Module
    {
    public:
        AMLines() = default;
        torch::Tensor forward(torch::Tensor x) const override;
    private:
    };




}



