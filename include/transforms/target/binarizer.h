#pragma once

#include "../common.h"


namespace xt::transforms::target
{
    class Binarizer final : public xt::Module
    {
    public:
        Binarizer();
        explicit Binarizer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
