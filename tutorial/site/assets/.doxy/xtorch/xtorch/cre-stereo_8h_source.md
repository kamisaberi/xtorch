

# File cre-stereo.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**stereo-matching-depth-estimation**](dir_e353cfd6010331702b3559c9641f7f23.md) **>** [**cre-stereo.h**](cre-stereo_8h.md)

[Go to the documentation of this file](cre-stereo_8h.md)


```C++
#pragma once


#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class CarlaStereo : public  BaseDataset {
    public :
        explicit  CarlaStereo(const std::string &root);
        CarlaStereo(const std::string &root, DataMode mode);
        CarlaStereo(const std::string &root, DataMode mode , bool download);
        CarlaStereo(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };


    class CREStereo : public BaseDataset {
    public :
        explicit  CREStereo(const std::string &root);
        CREStereo(const std::string &root, DataMode mode);
        CREStereo(const std::string &root, DataMode mode , bool download);
        CREStereo(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };
}
```


