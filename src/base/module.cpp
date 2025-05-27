#include "include/base/base.h"


namespace xt
{
    Module::Module() = default;


    // Operator() implementation in Module
    auto Module::operator()(const std::initializer_list<torch::Tensor> tensors) -> std::any
    {
        return forward(tensors); // Calls subclass's forward
    }
}
