#include "include/transforms/target/event_to_interval_converter.h"

namespace xt::transforms::target
{
    EventToIntervalConverter::EventToIntervalConverter() = default;

    EventToIntervalConverter::EventToIntervalConverter(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto EventToIntervalConverter::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
