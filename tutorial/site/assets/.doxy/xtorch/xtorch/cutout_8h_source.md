

# File cutout.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**transforms**](dir_de1d6215dd8b8d2c901daadc91a23b6e.md) **>** [**cutout.h**](cutout_8h.md)

[Go to the documentation of this file](cutout_8h.md)


```C++
#pragma once

#include "../headers/transforms.h"

namespace xt::data::transforms {


    struct Cutout {
    public:
        Cutout(int num_holes = 1, int hole_size = 16);

        torch::Tensor operator()(const torch::Tensor& input_tensor) const;
    private:
        int num_holes;
        int hole_size;

    };

}
```


