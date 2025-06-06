#pragma once

#include "../common.h"


namespace xt::transforms::signal
{
    class GriffinLim final : public xt::Module
    {
    public:
        GriffinLim();
        explicit GriffinLim(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
