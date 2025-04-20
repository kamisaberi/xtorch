

# File amazon-review.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**text-classification**](dir_50f41150f848aea77b9741968a6098a5.md) **>** [**amazon-review.h**](amazon-review_8h.md)

[Go to the documentation of this file](amazon-review_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class AmazonReview : BaseDataset {
        public :
            explicit AmazonReview(const std::string &root);
        AmazonReview(const std::string &root, DataMode mode);
        AmazonReview(const std::string &root, DataMode mode , bool download);
        AmazonReview(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


