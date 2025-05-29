#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class SentencePieceTokenizer final : public xt::Module
    {
    public:
        SentencePieceTokenizer();
        explicit SentencePieceTokenizer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
