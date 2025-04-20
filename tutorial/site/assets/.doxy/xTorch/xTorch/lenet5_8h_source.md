

# File lenet5.h

[**File List**](files.md) **>** [**cnn**](dir_40be95ab8912b8deac694fbe2f8f2654.md) **>** [**lenet**](dir_7143ffcf272660e648740977d1bb606b.md) **>** [**lenet5.h**](lenet5_8h.md)

[Go to the documentation of this file](lenet5_8h.md)


```C++
#pragma once
#include <torch/torch.h>
#include "../../base.h"
#include <iostream>
#include <vector>


using namespace std;

namespace xt::models {
    struct LeNet5 : BaseModel {
    protected:
        mutable  torch::nn::Sequential layer1 = nullptr, layer2 = nullptr;
        mutable  torch::nn::Linear fc1 = nullptr, fc2 = nullptr, fc3 = nullptr;

    public:
        LeNet5(int num_classes/* classes */, int in_channels = 1/*  input channels */);
        LeNet5(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };
}
```


