#include "include/transforms/image/channel_shuffle.h"

namespace xt::transforms::image {

    ChannelShuffle::ChannelShuffle() = default;

    ChannelShuffle::ChannelShuffle(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto ChannelShuffle::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}