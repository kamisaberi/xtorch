#pragma once

#include "../common.h"


namespace xt::transforms::target
{
    class LabelEncoder final : public xt::Module
    {
    public:
        LabelEncoder();
        explicit LabelEncoder(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
