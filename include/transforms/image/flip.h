#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class Flip final : public xt::Module
    {
    public:
        Flip();
        explicit Flip(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
