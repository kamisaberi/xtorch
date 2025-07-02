#pragma once

#include "../common.h"


namespace xt::transforms::text
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
