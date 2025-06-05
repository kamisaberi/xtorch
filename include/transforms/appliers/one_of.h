#pragma once

#include "../common.h"

namespace xt::transforms
{
    class OneOf final : public xt::Module
    {
    public:
        OneOf();
        explicit OneOf(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::vector<xt::Module> transforms;

    };
}
