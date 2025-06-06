#pragma once

#include "../common.h"


namespace xt::transforms::target
{
    class EventRateCalculator final : public xt::Module
    {
    public:
        EventRateCalculator();
        explicit EventRateCalculator(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
