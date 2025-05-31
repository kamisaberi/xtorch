#include "include/transforms/image/channel_shuffle.h"

namespace xt::transforms::image {

    ChannelShuffle::ChannelShuffle() = default;

    ChannelShuffle::ChannelShuffle(std::vector<xt::Module> transforms) : xt::Module() {
    }

    auto ChannelShuffle::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeros(10);
    }


}