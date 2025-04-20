

# File alexnet.h

[**File List**](files.md) **>** [**alexnet**](dir_3b7157f900ab20c97880c9a0f5c25c82.md) **>** [**alexnet.h**](alexnet_8h.md)

[Go to the documentation of this file](alexnet_8h.md)


```C++
#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "../../base.h"


using namespace std;


namespace xt::models {
    struct AlexNet : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr, layer5 =
                        nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

    public:
        AlexNet(int num_classes /* classes */, int in_channels = 3/* input channels */);

        AlexNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };
}
```


