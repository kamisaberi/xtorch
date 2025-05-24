#pragma once
#include "include/models/common.h"


using namespace std;

namespace xt::models {
    struct ResNeXt : xt::Cloneable<ResNeXt>
    {
    private:

    public:
        ResNeXt(int num_classes /* classes */, int in_channels = 3/* input channels */);

        ResNeXt(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


}
