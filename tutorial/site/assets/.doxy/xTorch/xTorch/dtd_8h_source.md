

# File dtd.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**image-classification**](dir_9d21d6f83a70094db43fe94b096ae893.md) **>** [**dtd.h**](dtd_8h.md)

[Go to the documentation of this file](dtd_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class DTD : public BaseDataset {
    public :
        explicit  DTD(const std::string &root);
        DTD(const std::string &root, DataMode mode);
        DTD(const std::string &root, DataMode mode , bool download);
        DTD(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


