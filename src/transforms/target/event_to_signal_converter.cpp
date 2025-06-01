#include "include/transforms/target/event_to_signal_converter.h"

namespace xt::transforms::target
{
    EventToSignalConverter::EventToSignalConverter() = default;

    EventToSignalConverter::EventToSignalConverter(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto EventToSignalConverter::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
