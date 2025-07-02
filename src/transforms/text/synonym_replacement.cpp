#include "include/transforms/text/synonym_replacement.h"

namespace xt::transforms::text
{
    SynonymReplacement::SynonymReplacement() = default;

    SynonymReplacement::SynonymReplacement(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto SynonymReplacement::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
