

# File lambda.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**transforms**](dir_de1d6215dd8b8d2c901daadc91a23b6e.md) **>** [**lambda.h**](lambda_8h.md)

[Go to the documentation of this file](lambda_8h.md)


```C++

#include "../headers/transforms.h"

namespace xt::data::transforms {

    struct Lambda {
    public:
        Lambda(std::function<torch::Tensor(torch::Tensor)> transform);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


}
```


