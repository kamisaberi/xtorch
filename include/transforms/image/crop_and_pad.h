#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class CropAndPad final : public xt::Module
    {
    public:
        CropAndPad();
        explicit CropAndPad(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
