#include "include/transforms/text/sentence_piece_tokenizer.h"

namespace xt::transforms::text
{
    SentencePieceTokenizer::SentencePieceTokenizer() = default;

    SentencePieceTokenizer::SentencePieceTokenizer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto SentencePieceTokenizer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
