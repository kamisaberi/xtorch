#pragma once

#include "../common.h"


namespace xt::transforms::signal
{
    class FrequencyMasking final : public xt::Module
    {
    public:
        FrequencyMasking();
        explicit FrequencyMasking(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
