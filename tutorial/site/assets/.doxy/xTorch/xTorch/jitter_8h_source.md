

# File jitter.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**transforms**](dir_de1d6215dd8b8d2c901daadc91a23b6e.md) **>** [**jitter.h**](jitter_8h.md)

[Go to the documentation of this file](jitter_8h.md)


```C++
#pragma once

#include "../headers/transforms.h"

namespace xt::data::transforms {


    struct ColorJitter {
    public:
        ColorJitter(float brightness = 0.0f,
                    float contrast = 0.0f,
                    float saturation = 0.0f);

        torch::Tensor operator()(const torch::Tensor& input_tensor) const;
    private:

        float brightness;
        float contrast;
        float saturation;

    };

}
```


