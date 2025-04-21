

# File compose.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**transforms**](dir_de1d6215dd8b8d2c901daadc91a23b6e.md) **>** [**compose.h**](compose_8h.md)

[Go to the documentation of this file](compose_8h.md)


```C++
#pragma once

#include "../headers/transforms.h"

namespace xt::data::transforms {

    class Compose {
    public:
        using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;

        Compose();

        Compose(std::vector<TransformFunc> transforms);

        torch::Tensor operator()(torch::Tensor input) const;

    private:
        std::vector<TransformFunc> transforms;
    };
}
```


