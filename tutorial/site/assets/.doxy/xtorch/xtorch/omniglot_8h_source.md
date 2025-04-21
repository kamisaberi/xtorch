

# File omniglot.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**image-classification**](dir_9d21d6f83a70094db43fe94b096ae893.md) **>** [**omniglot.h**](omniglot_8h.md)

[Go to the documentation of this file](omniglot_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class Omniglot : BaseDataset {
        /*
        https://github.com/brendenlake/omniglot
        """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
        Args:
            root (str or ``pathlib.Path``): Root directory of dataset where directory
                ``omniglot-py`` exists.
            background (bool, optional): If True, creates dataset from the "background" set, otherwise
                creates from the "evaluation" set. This terminology is defined by the authors.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            download (bool, optional): If true, downloads the dataset zip files from the internet and
                puts it in root directory. If the zip files are already downloaded, they are not
                downloaded again.
        """
         */
    public :
        explicit Omniglot(const std::string &root);
        Omniglot(const std::string &root, DataMode mode);
        Omniglot(const std::string &root, DataMode mode , bool download);
        Omniglot(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        // folder = "omniglot-py"
        // download_url_prefix = "https://raw.githubusercontent.com/brendenlake/omniglot/master/python"
        // zips_md5 = {
        // "images_background": "68d2efa1b9178cc56df9314c21c6e718",
        // "images_evaluation": "6b91aef0f799c5bb55b94e3f2daec811",
        // }

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
```


