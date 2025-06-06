#pragma once

#include "../common.h"


namespace xt::transforms::text
{
    class BackTranslation final : public xt::Module
    {
    public:
        BackTranslation();
        explicit BackTranslation(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
