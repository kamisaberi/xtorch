#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class CutMix final : public xt::Module
    {
    public:
        CutMix();
        explicit CutMix(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
