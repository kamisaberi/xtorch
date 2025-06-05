#pragma once

#include "../common.h"


namespace xt::transforms::weather
{
    class PatchyFog final : public xt::Module
    {
    public:
        PatchyFog();
        explicit PatchyFog(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
