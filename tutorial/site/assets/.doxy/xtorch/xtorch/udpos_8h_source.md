

# File udpos.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**sequence-tagging**](dir_c4dca688f613c914aa3d806c1f628e0e.md) **>** [**udpos.h**](udpos_8h.md)

[Go to the documentation of this file](udpos_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class UDPOS : BaseDataset {
        public :
            explicit UDPOS(const std::string &root);
        UDPOS(const std::string &root, DataMode mode);
        UDPOS(const std::string &root, DataMode mode , bool download);
        UDPOS(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


