#include "include/transforms/text/text_style_transfer.h"

namespace xt::transforms::text
{
    TextStyleTransfer::TextStyleTransfer() = default;

    TextStyleTransfer::TextStyleTransfer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto TextStyleTransfer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
