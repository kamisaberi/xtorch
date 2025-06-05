#pragma once

#include "../common.h"


namespace xt::transforms::signal
{
    class SlidingWindowCMN final : public xt::Module
    {
    public:
        SlidingWindowCMN();
        explicit SlidingWindowCMN(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
