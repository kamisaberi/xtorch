#pragma once

#include "../common.h"


namespace xt::transforms::signal
{
    class MFCC final : public xt::Module
    {
    public:
        MFCC();
        explicit MFCC(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
