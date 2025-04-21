

# File cmu-arctic.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**cmu-arctic.h**](cmu-arctic_8h.md)

[Go to the documentation of this file](cmu-arctic_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class CMUArctic : BaseDataset {
    public :
        explicit CMUArctic(const std::string &root);
        CMUArctic(const std::string &root, DataMode mode);
        CMUArctic(const std::string &root, DataMode mode , bool download);
        CMUArctic(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


