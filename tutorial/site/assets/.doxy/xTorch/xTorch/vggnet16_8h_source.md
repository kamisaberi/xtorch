

# File vggnet16.h

[**File List**](files.md) **>** [**cnn**](dir_40be95ab8912b8deac694fbe2f8f2654.md) **>** [**vggnet**](dir_ef7a937debe30c3ca367a3d686ce12d7.md) **>** [**vggnet16.h**](vggnet16_8h.md)

[Go to the documentation of this file](vggnet16_8h.md)


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
    struct VggNet16 : BaseModel {
        mutable torch::nn::Sequential layer1 = nullptr, layer2 = nullptr, layer3 = nullptr, layer4 = nullptr;
        mutable torch::nn::Sequential layer5 = nullptr;
        mutable torch::nn::Sequential layer6 = nullptr, layer7 = nullptr, layer8 = nullptr, layer9 = nullptr;
        mutable torch::nn::Sequential layer10 = nullptr;
        mutable torch::nn::Sequential layer11 = nullptr, layer12 = nullptr, layer13 = nullptr;
        mutable torch::nn::Sequential fc = nullptr, fc1 = nullptr, fc2 = nullptr;

        VggNet16(int num_classes /* classes */, int in_channels  /* input channels */);
        VggNet16(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };
}
```


