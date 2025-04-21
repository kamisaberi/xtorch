

# File common-voice.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**common-voice.h**](common-voice_8h.md)

[Go to the documentation of this file](common-voice_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class CommonVoice : BaseDataset {
    public :
        explicit CommonVoice(const std::string &root);
        CommonVoice(const std::string &root, DataMode mode);
        CommonVoice(const std::string &root, DataMode mode , bool download);
        CommonVoice(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


