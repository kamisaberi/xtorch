#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Compose final : public xt::Module
    {
    public:
        Compose();
        explicit Compose(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<torch::Tensor> tensors) -> std::any override;

    private:
        std::vector<xt::Module> transforms;
    };
}
