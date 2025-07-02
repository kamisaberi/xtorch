#pragma once

#include "../common.h"


namespace xt::transforms::target
{
    class LogTransformer final : public xt::Module
    {
    public:
        LogTransformer();
        explicit LogTransformer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
