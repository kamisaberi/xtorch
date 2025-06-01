#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::signal
{
    class PitchShift final : public xt::Module
    {
    public:
        PitchShift();
        explicit PitchShift(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
