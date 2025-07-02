#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class OpticalDistortion final : public xt::Module
    {
    public:
        OpticalDistortion();
        explicit OpticalDistortion(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
