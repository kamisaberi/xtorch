#include "include/transforms/target/label_noise_injector.h"

namespace xt::transforms::target
{
    LabelNoiseInjector::LabelNoiseInjector() = default;

    LabelNoiseInjector::LabelNoiseInjector(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto LabelNoiseInjector::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
