#include "include/transforms/image/channel_dropout.h"

namespace xt::transforms::image {

    ChannelDropout::ChannelDropout() = default;

    ChannelDropout::ChannelDropout(std::function<torch::Tensor(torch::Tensor)> transform) : xt::Module(), transform(transform) {
    }

    auto ChannelDropout::forward(std::initializer_list <std::any> tensors) -> std::any {
        return torch::zeroes(10);
    }


}