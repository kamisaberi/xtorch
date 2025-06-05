#pragma once

#include "../common.h"


namespace xt::transforms::weather
{
    class Blizzard final : public xt::Module
    {
    public:
        Blizzard();
        explicit Blizzard(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
