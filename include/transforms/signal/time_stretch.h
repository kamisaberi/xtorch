#pragma once

#include "../common.h"


namespace xt::transforms::signal
{
    class TimeStretch final : public xt::Module
    {
    public:
        TimeStretch();
        explicit TimeStretch(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
