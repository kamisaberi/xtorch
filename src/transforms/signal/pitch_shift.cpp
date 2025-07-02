#include "include/transforms/signal/pitch_shift.h"

namespace xt::transforms::signal
{
    PitchShift::PitchShift() = default;

    PitchShift::PitchShift(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto PitchShift::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
