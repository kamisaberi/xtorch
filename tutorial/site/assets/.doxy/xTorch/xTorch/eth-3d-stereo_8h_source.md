

# File eth-3d-stereo.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**stereo-matching-depth-estimation**](dir_e353cfd6010331702b3559c9641f7f23.md) **>** [**eth-3d-stereo.h**](eth-3d-stereo_8h.md)

[Go to the documentation of this file](eth-3d-stereo_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class ETH3DStereo : public BaseDataset {
    public:
        explicit ETH3DStereo(const std::string &root);
        ETH3DStereo(const std::string &root, DataMode mode);
        ETH3DStereo(const std::string &root, DataMode mode , bool download);
        ETH3DStereo(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };
}
```


