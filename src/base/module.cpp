#include "include/base/base.h"


namespace xt
{
    Module::Module() = default;

    torch::Tensor Module::operator()(torch::Tensor input) const
    {
        return this->forward(std::move(input));
    }
}
