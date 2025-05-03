#pragma once

namespace xt::transforms::image
{
    struct CenterCrop {
    public:
        CenterCrop(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::vector<int64_t> size;
    };

}