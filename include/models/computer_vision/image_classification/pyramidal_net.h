#pragma once

#include "models/common.h"


namespace xt::models
{
    struct PyramidalNet : xt::Cloneable<PyramidalNet>
    {
    private:

    public:
        PyramidalNet(int num_classes /* classes */, int in_channels = 3/* input channels */);

        PyramidalNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}