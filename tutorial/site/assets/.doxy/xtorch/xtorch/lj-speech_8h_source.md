

# File lj-speech.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**lj-speech.h**](lj-speech_8h.md)

[Go to the documentation of this file](lj-speech_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class LjSpeech : BaseDataset {
    public :
        explicit LjSpeech(const std::string &root);

        LjSpeech(const std::string &root, DataMode mode);

        LjSpeech(const std::string &root, DataMode mode, bool download);

        LjSpeech(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


