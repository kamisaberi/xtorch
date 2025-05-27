#include "include/base/module1.h"


namespace xt
{
    Module1::Module1() = default;

    // Operator() implementation in Module
    auto Module1::operator()(const std::initializer_list<torch::Tensor> tensors) -> std::any
    {
        return forward(tensors); // Calls subclass's forward
    }
}
