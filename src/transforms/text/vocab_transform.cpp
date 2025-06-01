#include "include/transforms/text/vocab_transform.h"

namespace xt::transforms::text
{
    VocabTransform::VocabTransform() = default;

    VocabTransform::VocabTransform(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto VocabTransform::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
