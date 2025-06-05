#pragma once

#include "../common.h"


namespace xt::transforms::signal
{
    class Vol final : public xt::Module
    {
    public:
        Vol();
        explicit Vol(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
