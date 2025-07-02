#include "include/transforms/target/event_rate_calculator.h"

namespace xt::transforms::target
{
    EventRateCalculator::EventRateCalculator() = default;

    EventRateCalculator::EventRateCalculator(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto EventRateCalculator::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
