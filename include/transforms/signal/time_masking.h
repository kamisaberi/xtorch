#pragma once

#include "../common.h"


namespace xt::transforms::signal
{
    class TimeMasking final : public xt::Module
    {
    public:
        TimeMasking();
        explicit TimeMasking(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
