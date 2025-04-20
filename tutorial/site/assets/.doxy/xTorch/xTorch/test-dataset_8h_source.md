

# File test-dataset.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**temp**](dir_9b1ced676d335869719a7bfb99e201c3.md) **>** [**test-dataset.h**](test-dataset_8h.md)

[Go to the documentation of this file](test-dataset_8h.md)


```C++
#pragma once

#include <torch/torch.h>

namespace xt::temp {

class TestDataset {

  public:
    TestDataset();
//    ~TestDataset();
    torch::Tensor forward(torch::Tensor input);
};

}

```


