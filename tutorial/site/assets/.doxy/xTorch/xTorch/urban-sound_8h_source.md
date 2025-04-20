

# File urban-sound.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**urban-sound.h**](urban-sound_8h.md)

[Go to the documentation of this file](urban-sound_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class UrbanSound : BaseDataset {
    public :
        explicit UrbanSound(const std::string &root);

        UrbanSound(const std::string &root, DataMode mode);

        UrbanSound(const std::string &root, DataMode mode, bool download);

        UrbanSound(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


