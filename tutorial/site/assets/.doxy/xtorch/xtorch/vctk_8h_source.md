

# File vctk.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**vctk.h**](vctk_8h.md)

[Go to the documentation of this file](vctk_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class VCTK : BaseDataset {
        public :
            explicit VCTK(const std::string &root);
        VCTK(const std::string &root, DataMode mode);
        VCTK(const std::string &root, DataMode mode , bool download);
        VCTK(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


