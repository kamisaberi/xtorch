

# File functions.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**definitions**](dir_92610beb03a79c6b2c32aee384ba5d92.md) **>** [**functions.h**](functions_8h.md)

[Go to the documentation of this file](functions_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data {
    torch::data::datasets::MapDataset<xt::data::datasets::BaseDataset , torch::data::transforms::Stack<>()>
    transform_dataset(xt::data::datasets::BaseDataset dataset, vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms);
}
```


