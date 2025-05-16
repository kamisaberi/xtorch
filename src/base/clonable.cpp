#include "../../include/base/cloneable.h"

namespace xt
{
    CloneableModule::CloneableModule()
    {
        reset();
    }

    torch::Tensor CloneableModule::operator()(torch::Tensor input) const
    {
        return this->forward(std::move(input));
    }
    void CloneableModule::reset() {
        // Linear layer is already initialized in the constructor
    }
}
