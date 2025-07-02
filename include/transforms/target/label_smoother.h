#pragma once

#include "../common.h"


namespace xt::transforms::target
{
    class LabelSmoother final : public xt::Module
    {
    public:
        LabelSmoother();
        explicit LabelSmoother(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
