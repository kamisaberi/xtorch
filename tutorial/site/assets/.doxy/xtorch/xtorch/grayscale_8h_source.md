

# File grayscale.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**transforms**](dir_de1d6215dd8b8d2c901daadc91a23b6e.md) **>** [**grayscale.h**](grayscale_8h.md)

[Go to the documentation of this file](grayscale_8h.md)


```C++
#pragma once



#include "../headers/transforms.h"

namespace xt::data::transforms {



    struct GrayscaleToRGB {
    public:
        torch::Tensor operator()(const torch::Tensor &tensor);
    };

    struct Grayscale {
    public:
        Grayscale();

        torch::Tensor operator()(torch::Tensor input);
    };

    struct ToGray {
        torch::Tensor operator()(const torch::Tensor& color_tensor) const;
    };


}
```


