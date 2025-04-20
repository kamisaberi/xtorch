

# File librispeech.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**librispeech.h**](librispeech_8h.md)

[Go to the documentation of this file](librispeech_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class LibriSpeech : BaseDataset {
    public :
        explicit LibriSpeech(const std::string &root);

        LibriSpeech(const std::string &root, DataMode mode);

        LibriSpeech(const std::string &root, DataMode mode, bool download);

        LibriSpeech(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


