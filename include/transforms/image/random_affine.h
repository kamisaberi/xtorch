#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class RandomAffine final : public xt::Module
    {
    public:
        RandomAffine();
        explicit RandomAffine(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
