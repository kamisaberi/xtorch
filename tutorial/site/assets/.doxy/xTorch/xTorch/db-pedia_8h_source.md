

# File db-pedia.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**text-classification**](dir_50f41150f848aea77b9741968a6098a5.md) **>** [**db-pedia.h**](db-pedia_8h.md)

[Go to the documentation of this file](db-pedia_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class DBPedia : BaseDataset {
        public :
            explicit DBPedia(const std::string &root);
        DBPedia(const std::string &root, DataMode mode);
        DBPedia(const std::string &root, DataMode mode , bool download);
        DBPedia(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


