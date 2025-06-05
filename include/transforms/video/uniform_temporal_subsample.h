#pragma once

#include "../common.h"


namespace xt::transforms::video
{
    class UniformTemporalSubsample final : public xt::Module
    {
    public:
        UniformTemporalSubsample();
        explicit UniformTemporalSubsample(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
