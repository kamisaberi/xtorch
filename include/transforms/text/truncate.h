#pragma once

#include "../common.h"


namespace xt::transforms::text
{
    class Truncate final : public xt::Module
    {
    public:
        Truncate();
        explicit Truncate(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
