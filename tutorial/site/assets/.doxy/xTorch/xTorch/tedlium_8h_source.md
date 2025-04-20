

# File tedlium.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**tedlium.h**](tedlium_8h.md)

[Go to the documentation of this file](tedlium_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class Tedlium : BaseDataset {
    public :
        explicit Tedlium(const std::string &root);

        Tedlium(const std::string &root, DataMode mode);

        Tedlium(const std::string &root, DataMode mode, bool download);

        Tedlium(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


