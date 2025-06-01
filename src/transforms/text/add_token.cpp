#include "include/transforms/text/add_token.h"

namespace xt::transforms::text
{
    AddToken::AddToken() = default;

    AddToken::AddToken(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto AddToken::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
