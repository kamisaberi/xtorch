

# File esc.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**esc.h**](esc_8h.md)

[Go to the documentation of this file](esc_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class ESC : BaseDataset {
    public :
        explicit ESC(const std::string &root);

        ESC(const std::string &root, DataMode mode);

        ESC(const std::string &root, DataMode mode, bool download);

        ESC(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


