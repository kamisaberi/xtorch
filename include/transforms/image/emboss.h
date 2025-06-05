#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class Emboss final : public xt::Module
    {
    public:
        Emboss();
        explicit Emboss(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
