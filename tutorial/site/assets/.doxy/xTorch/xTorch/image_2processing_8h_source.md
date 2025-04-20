

# File processing.h

[**File List**](files.md) **>** [**image**](dir_b86c9d22e47aa9431a2bbc5d6808d12b.md) **>** [**processing.h**](image_2processing_8h.md)

[Go to the documentation of this file](image_2processing_8h.md)


```C++
#pragma once

#include <iostream>
#include <torch/torch.h>


namespace torch::ext::media::image {
    torch::Tensor resize(const torch::Tensor &tensor, const std::vector<int64_t> &size);
}
```


