#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class ADA final : public xt::Module
    {
    public:
        ADA();
        explicit ADA(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
