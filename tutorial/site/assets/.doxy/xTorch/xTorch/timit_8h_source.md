

# File timit.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**timit.h**](timit_8h.md)

[Go to the documentation of this file](timit_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class TIMIT : BaseDataset {
    public :
        explicit TIMIT(const std::string &root);

        TIMIT(const std::string &root, DataMode mode);

        TIMIT(const std::string &root, DataMode mode, bool download);

        TIMIT(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


