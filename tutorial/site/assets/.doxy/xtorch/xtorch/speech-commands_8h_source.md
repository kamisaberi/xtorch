

# File speech-commands.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**speech-commands.h**](speech-commands_8h.md)

[Go to the documentation of this file](speech-commands_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class SpeechCommands : BaseDataset {
    public :
        explicit SpeechCommands(const std::string &root);

        SpeechCommands(const std::string &root, DataMode mode);

        SpeechCommands(const std::string &root, DataMode mode, bool download);

        SpeechCommands(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


