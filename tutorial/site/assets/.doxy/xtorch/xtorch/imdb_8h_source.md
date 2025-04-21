

# File imdb.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**sentiment-analysis**](dir_2c5adc0a1688a9a1937194391138274f.md) **>** [**imdb.h**](imdb_8h.md)

[Go to the documentation of this file](imdb_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class IMDB : BaseDataset {
        public :
            explicit IMDB(const std::string &root);
        IMDB(const std::string &root, DataMode mode);
        IMDB(const std::string &root, DataMode mode , bool download);
        IMDB(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


