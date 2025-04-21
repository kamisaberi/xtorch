

# File usps.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**image-classification**](dir_9d21d6f83a70094db43fe94b096ae893.md) **>** [**usps.h**](usps_8h.md)

[Go to the documentation of this file](usps_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class USPS : BaseDataset {
        /*
        """`USPS <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps>`_ Dataset.
    The data-format is : [label [index:value ]*256 \\n] * num_lines, where ``label`` lies in ``[1, 10]``.
    The value for each pixel lies in ``[-1, 1]``. Here we transform the ``label`` into ``[0, 9]``
    and make pixel values in ``[0, 255]``.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset to store``USPS`` data files.
        train (bool, optional): If True, creates dataset from ``usps.bz2``,
            otherwise from ``usps.t.bz2``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

         */
    public :
        explicit USPS(const std::string &root);
        USPS(const std::string &root, DataMode mode);
        USPS(const std::string &root, DataMode mode , bool download);
        USPS(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        std::map<std::string, std::tuple<fs::path, fs::path, std::string>> resources = {
                {"train", {
                                  fs::path(
                                          "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2"),
                                  fs::path("usps.bz2"),
                                  "ec16c51db3855ca6c91edd34d0e9b197"
                          }

                },
                {"test",  {
                                  fs::path(
                                          "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2"),
                                  fs::path("usps.t.bz2"),
                                  "8ea070ee2aca1ac39742fdd1ef5ed118"
                          }
                }
        };
        void load_data();

        void check_resources();

    };
}
```


