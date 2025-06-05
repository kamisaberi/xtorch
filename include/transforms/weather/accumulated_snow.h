#pragma once

#include "../common.h"


namespace xt::transforms::weather
{
    class AccumulatedSnow final : public xt::Module
    {
    public:
        AccumulatedSnow();
        explicit AccumulatedSnow(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
