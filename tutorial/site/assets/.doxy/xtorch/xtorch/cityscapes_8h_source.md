

# File cityscapes.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**object-detection-and-segmentation**](dir_6e95aff3cb8ce7a70a5f1e2f7dd69202.md) **>** [**cityscapes.h**](cityscapes_8h.md)

[Go to the documentation of this file](cityscapes_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class Cityscapes : public BaseDataset {
    public :
        explicit  Cityscapes(const std::string &root);
        Cityscapes(const std::string &root, DataMode mode);
        Cityscapes(const std::string &root, DataMode mode , bool download);
        Cityscapes(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


