

# File imagenet.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**image-classification**](dir_9d21d6f83a70094db43fe94b096ae893.md) **>** [**imagenet.h**](imagenet_8h.md)

[Go to the documentation of this file](imagenet_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class ImageNet : BaseDataset {
    public :
        explicit  ImageNet(const std::string &root);
        ImageNet(const std::string &root, DataMode mode);
        ImageNet(const std::string &root, DataMode mode , bool download);
        ImageNet(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };
}
```


