

# File resize.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**transforms**](dir_de1d6215dd8b8d2c901daadc91a23b6e.md) **>** [**resize.h**](resize_8h.md)

[Go to the documentation of this file](resize_8h.md)


```C++
#pragma once


#include "../headers/transforms.h"

namespace xt::data::transforms {


    struct Resize {
    public:
        Resize(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor img);

    private:
        std::vector<int64_t> size; 
    };




}
```


