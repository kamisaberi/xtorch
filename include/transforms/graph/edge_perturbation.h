#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class EdgePerturbation final : public xt::Module
    {
    public:
        EdgePerturbation();
        explicit EdgePerturbation(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
