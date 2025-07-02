#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class Spatter final : public xt::Module
    {
    public:
        Spatter();
        explicit Spatter(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
