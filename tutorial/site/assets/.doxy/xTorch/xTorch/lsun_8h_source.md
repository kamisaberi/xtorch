

# File lsun.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**specific**](dir_e5ef08163bed877f164b8cca216875b1.md) **>** [**lsun.h**](lsun_8h.md)

[Go to the documentation of this file](lsun_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    [[deprecated("LSUN Dataset files removed and Links are broken")]]
    class LSUN  : BaseDataset  {
    public :
        LSUN(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        LSUN(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
```


