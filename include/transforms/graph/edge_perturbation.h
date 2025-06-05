#pragma once

#include "../common.h"


namespace xt::transforms::graph
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
