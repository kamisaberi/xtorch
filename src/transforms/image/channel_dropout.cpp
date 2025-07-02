#include "include/transforms/image/channel_dropout.h"

namespace xt::transforms::image {

    ChannelDropout::ChannelDropout() = default;

    ChannelDropout::ChannelDropout(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto ChannelDropout::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}