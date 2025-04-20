

# File penn-treebank.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**language-modeling**](dir_2ee0048eab60d09605b89e5e753a33b4.md) **>** [**penn-treebank.h**](penn-treebank_8h.md)

[Go to the documentation of this file](penn-treebank_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class PennTreebank : BaseDataset {
        public :
            explicit PennTreebank(const std::string &root);
        PennTreebank(const std::string &root, DataMode mode);
        PennTreebank(const std::string &root, DataMode mode , bool download);
        PennTreebank(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
```


