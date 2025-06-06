#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomPerspective final : public xt::Module
    {
    public:
        RandomPerspective();
        explicit RandomPerspective(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
