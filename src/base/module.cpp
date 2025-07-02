#include "include/base/base.h"


namespace xt
{
    Module::Module() = default;
    // Operator() implementation in Module
    auto Module::operator()(std::initializer_list<std::any> tensors) -> std::any
    {
        return forward(tensors); // Calls subclass's forward
    }
}
