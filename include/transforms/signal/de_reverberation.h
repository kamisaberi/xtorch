#pragma once

#include "../common.h"


namespace xt::transforms::signal
{
    class DeReverberation final : public xt::Module
    {
    public:
        DeReverberation();
        explicit DeReverberation(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
