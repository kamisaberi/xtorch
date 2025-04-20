

# File normalize.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**transforms**](dir_de1d6215dd8b8d2c901daadc91a23b6e.md) **>** [**normalize.h**](normalize_8h.md)

[Go to the documentation of this file](normalize_8h.md)


```C++

#include "../headers/transforms.h"

namespace xt::data::transforms {
    struct Normalize {
    public:
        Normalize(std::vector<float> mean, std::vector<float> std);
        torch::Tensor operator()(const torch::Tensor& tensor) const;
    private:
        std::vector<float> mean;
        std::vector<float> std;

    };

}
```


