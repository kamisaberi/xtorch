#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class Equalize final : public xt::Module
    {
    public:
        Equalize();
        explicit Equalize(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
