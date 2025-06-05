#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class Affine final : public xt::Module
    {
    public:
        Affine();
        explicit Affine(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
