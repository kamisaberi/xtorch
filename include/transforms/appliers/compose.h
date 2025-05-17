#pragma once

#include "transforms/common.h"

namespace xt::transforms
{
    class Compose final : public xt::Module
    {
    public:
        Compose();

        explicit Compose(std::vector<xt::Module> transforms);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::vector<xt::Module> transforms;

    };
}
