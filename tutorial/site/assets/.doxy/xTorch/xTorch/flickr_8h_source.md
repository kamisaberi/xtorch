

# File flickr.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**image-classification**](dir_9d21d6f83a70094db43fe94b096ae893.md) **>** [**flickr.h**](flickr_8h.md)

[Go to the documentation of this file](flickr_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    [[deprecated("Flickr8k Dataset some files removed and Links are broken")]]
    class Flickr8k :public BaseDataset {
    public :
        Flickr8k(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Flickr8k(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    [[deprecated("Flickr30k Dataset some files removed and Links are broken")]]
    class Flickr30k :public BaseDataset {
    public :
        Flickr30k(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Flickr30k(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
```


