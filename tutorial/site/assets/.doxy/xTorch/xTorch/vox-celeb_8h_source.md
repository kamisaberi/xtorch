

# File vox-celeb.h

[**File List**](files.md) **>** [**audio-speech**](dir_3f959236e5b642d039994a38a6e55324.md) **>** [**vox-celeb.h**](vox-celeb_8h.md)

[Go to the documentation of this file](vox-celeb_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class VoxCeleb : BaseDataset {
        public :
            explicit VoxCeleb(const std::string &root);
        VoxCeleb(const std::string &root, DataMode mode);
        VoxCeleb(const std::string &root, DataMode mode , bool download);
        VoxCeleb(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


