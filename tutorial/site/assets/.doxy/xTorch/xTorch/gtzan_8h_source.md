

# File gtzan.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**gtzan.h**](gtzan_8h.md)

[Go to the documentation of this file](gtzan_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class GTZAN : BaseDataset {
    public :
        explicit GTZAN(const std::string &root);

        GTZAN(const std::string &root, DataMode mode);

        GTZAN(const std::string &root, DataMode mode, bool download);

        GTZAN(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


