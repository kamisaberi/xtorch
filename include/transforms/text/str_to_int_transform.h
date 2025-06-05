#pragma once

#include "../common.h"


namespace xt::transforms::text
{
    class StrToIntTransform final : public xt::Module
    {
    public:
        StrToIntTransform();
        explicit StrToIntTransform(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
