

# File sun.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**image-classification**](dir_9d21d6f83a70094db43fe94b096ae893.md) **>** [**sun.h**](sun_8h.md)

[Go to the documentation of this file](sun_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class SUN397 : BaseDataset {
        /*
        """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.

    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

         */
    public :
        explicit SUN397(const std::string &root);
        SUN397(const std::string &root, DataMode mode);
        SUN397(const std::string &root, DataMode mode , bool download);
        SUN397(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        fs::path url = fs::path("http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz");
        fs::path dataset_file_name = "SUN397.tar.gz";
        std::string dataset_file_md5 = "8ca2778205c41d23104230ba66911c7a";
        fs::path dataset_folder_name = "sun397";

        void load_data();


        void check_resources();

    };
}
```


