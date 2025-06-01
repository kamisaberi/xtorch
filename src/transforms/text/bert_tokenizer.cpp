#include "include/transforms/text/bert_tokenizer.h"

namespace xt::transforms::text
{
    BertTokenizer::BertTokenizer() = default;

    BertTokenizer::BertTokenizer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto BertTokenizer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
