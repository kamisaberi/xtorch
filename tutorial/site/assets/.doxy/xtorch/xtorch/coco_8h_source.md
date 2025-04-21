

# File coco.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**object-detection-and-segmentation**](dir_6e95aff3cb8ce7a70a5f1e2f7dd69202.md) **>** [**coco.h**](coco_8h.md)

[Go to the documentation of this file](coco_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class CocoDetection : public BaseDataset {
    public :
        explicit  CocoDetection(const std::string &root);
        CocoDetection(const std::string &root, DataMode mode);
        CocoDetection(const std::string &root, DataMode mode , bool download);
        CocoDetection(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };

    class CocoCaptions : public BaseDataset {
    public :
        explicit  CocoCaptions(const std::string &root);
        CocoCaptions(const std::string &root, DataMode mode);
        CocoCaptions(const std::string &root, DataMode mode , bool download);
        CocoCaptions(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
```


