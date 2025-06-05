#pragma once

#include "../common.h"


namespace xt::transforms::signal
{
    class MelSpectrogram final : public xt::Module
    {
    public:
        MelSpectrogram();
        explicit MelSpectrogram(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
