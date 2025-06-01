#include "include/transforms/target/leave_one_out_encoder.h"

namespace xt::transforms::target
{
    LeaveOneOutEncoder::LeaveOneOutEncoder() = default;

    LeaveOneOutEncoder::LeaveOneOutEncoder(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto LeaveOneOutEncoder::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
