

# File transforms.h

[**File List**](files.md) **>** [**definitions**](dir_11d78a78cbacc94abc067fbb8f3d9498.md) **>** [**transforms.h**](definitions_2transforms_8h.md)

[Go to the documentation of this file](definitions_2transforms_8h.md)


```C++
#pragma once

#include "../headers/transforms.h"


namespace xt::data::transforms {
    std::function<torch::Tensor(torch::Tensor input)> create_resize_transform(std::vector<int64_t> size);

    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size);

    torch::data::transforms::Lambda<torch::data::Example<> > resize(std::vector<int64_t> size);

    torch::data::transforms::Lambda<torch::data::Example<> > normalize(double mean, double stddev);





















}
```


