

# File qnli.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**natural-language-inference**](dir_cecfbd08ba907cb0c98c6ffe5c1549f6.md) **>** [**qnli.h**](qnli_8h.md)

[Go to the documentation of this file](qnli_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class QNLI : BaseDataset {
        public :
            explicit QNLI(const std::string &root);
        QNLI(const std::string &root, DataMode mode);
        QNLI(const std::string &root, DataMode mode , bool download);
        QNLI(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


