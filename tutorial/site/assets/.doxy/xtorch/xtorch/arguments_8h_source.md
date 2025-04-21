

# File arguments.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**types**](dir_0ad255a918b7fba820a1ddafed6fa637.md) **>** [**arguments.h**](arguments_8h.md)

[Go to the documentation of this file](arguments_8h.md)


```C++
#pragma once
#include <vector>
#include <torch/data.h>
#include "enums.h"
using namespace std;


struct DatasetArguments {
    DataMode mode = DataMode::TRAIN;
    bool download = false;
    vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms  = {};

};
```


