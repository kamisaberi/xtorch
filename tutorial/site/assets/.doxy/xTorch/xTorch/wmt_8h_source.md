

# File wmt.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**machine-translation**](dir_91ef7b38e4721f67f17b6805785cb95a.md) **>** [**wmt.h**](wmt_8h.md)

[Go to the documentation of this file](wmt_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class WMT : BaseDataset {
        public :
            explicit WMT(const std::string &root);
        WMT(const std::string &root, DataMode mode);
        WMT(const std::string &root, DataMode mode , bool download);
        WMT(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


