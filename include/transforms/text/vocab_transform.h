#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class VocabTransform final : public xt::Module
    {
    public:
        VocabTransform();
        explicit VocabTransform(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
