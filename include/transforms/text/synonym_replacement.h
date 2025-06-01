#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::text
{
    class SynonymReplacement final : public xt::Module
    {
    public:
        SynonymReplacement();
        explicit SynonymReplacement(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
