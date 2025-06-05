#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class CropNonEmptyMaskIfExists final : public xt::Module
    {
    public:
        CropNonEmptyMaskIfExists();
        explicit CropNonEmptyMaskIfExists(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
