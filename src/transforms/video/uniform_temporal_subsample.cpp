#include "include/transforms/video/uniform_temporal_subsample.h"

namespace xt::transforms::video
{
    UniformTemporalSubsample::UniformTemporalSubsample() = default;

    UniformTemporalSubsample::UniformTemporalSubsample(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto UniformTemporalSubsample::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
