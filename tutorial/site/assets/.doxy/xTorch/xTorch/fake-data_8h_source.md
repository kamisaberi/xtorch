

# File fake-data.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**image-classification**](dir_9d21d6f83a70094db43fe94b096ae893.md) **>** [**fake-data.h**](fake-data_8h.md)

[Go to the documentation of this file](fake-data_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class FakeData : public BaseDataset {
    public :
        FakeData();

        FakeData(size_t size);

        FakeData( size_t size, vector<int64_t> shape);

        FakeData( size_t size, vector<int64_t> shape, TransformType transforms);

    private :
        size_t size_;
        vector<int64_t> shape_ = {3, 24, 24};

        void generate_data();
    };
}
```


