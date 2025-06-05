#pragma once

#include "../common.h"


namespace xt::transforms::signal
{
    class WaveletTransforms final : public xt::Module
    {
    public:
        WaveletTransforms();
        explicit WaveletTransforms(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
