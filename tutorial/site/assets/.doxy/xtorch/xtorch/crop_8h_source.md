

# File crop.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**transforms**](dir_de1d6215dd8b8d2c901daadc91a23b6e.md) **>** [**crop.h**](crop_8h.md)

[Go to the documentation of this file](crop_8h.md)


```C++
#pragma once

#include "../headers/transforms.h"

namespace xt::data::transforms {


    struct CenterCrop {
    public:
        CenterCrop(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::vector<int64_t> size;
    };


    struct RandomCrop {
    public:
        RandomCrop(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::vector<int64_t> size;
    };


    struct RandomCrop2 {
    private:
        int crop_height;
        int crop_width;

    public:
        RandomCrop2(int height, int width);

        torch::Tensor operator()(const torch::Tensor &input_tensor);
    };





}
```


