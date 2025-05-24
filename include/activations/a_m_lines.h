#pragma once

#include "common.h"

namespace xt::activations
{
    struct AMLines : xt::Module
    {
    public:
        AMLines() = default;
        torch::Tensor forward(torch::Tensor x) const override;
    private:
    };


    torch::Tensor am_lines(torch::Tensor x);

}



