#pragma once

#include "../common.h"


namespace xt::transforms::target
{
    class LabelRemapper final : public xt::Module
    {
    public:
        LabelRemapper();
        explicit LabelRemapper(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
