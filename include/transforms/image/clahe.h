#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class CLAHE final : public xt::Module
    {
    public:
        CLAHE();
        explicit CLAHE(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
