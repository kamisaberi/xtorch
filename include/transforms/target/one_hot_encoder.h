#pragma once

#include "../common.h"


namespace xt::transforms::target
{
    class OneHotEncoder final : public xt::Module
    {
    public:
        OneHotEncoder();
        explicit OneHotEncoder(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
