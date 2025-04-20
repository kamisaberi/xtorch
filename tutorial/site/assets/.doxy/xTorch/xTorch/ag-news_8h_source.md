

# File ag-news.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**text-classification**](dir_50f41150f848aea77b9741968a6098a5.md) **>** [**ag-news.h**](ag-news_8h.md)

[Go to the documentation of this file](ag-news_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class AgNews : BaseDataset {
        public :
            explicit AgNews(const std::string &root);
        AgNews(const std::string &root, DataMode mode);
        AgNews(const std::string &root, DataMode mode , bool download);
        AgNews(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


