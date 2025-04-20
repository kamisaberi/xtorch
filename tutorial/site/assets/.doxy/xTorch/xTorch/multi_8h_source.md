

# File multi.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**machine-translation**](dir_91ef7b38e4721f67f17b6805785cb95a.md) **>** [**multi.h**](multi_8h.md)

[Go to the documentation of this file](multi_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class MULTI : BaseDataset {
        public :
            explicit MULTI(const std::string &root);
        MULTI(const std::string &root, DataMode mode);
        MULTI(const std::string &root, DataMode mode , bool download);
        MULTI(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


