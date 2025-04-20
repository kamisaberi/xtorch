

# File con-ll-2000-chunking.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**sequence-tagging**](dir_c4dca688f613c914aa3d806c1f628e0e.md) **>** [**con-ll-2000-chunking.h**](con-ll-2000-chunking_8h.md)

[Go to the documentation of this file](con-ll-2000-chunking_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class CoNLL2000Chunking : BaseDataset {
        public :
            explicit CoNLL2000Chunking(const std::string &root);
        CoNLL2000Chunking(const std::string &root, DataMode mode);
        CoNLL2000Chunking(const std::string &root, DataMode mode , bool download);
        CoNLL2000Chunking(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


