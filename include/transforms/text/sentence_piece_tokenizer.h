#pragma once

#include "../common.h"


namespace xt::transforms::text
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
