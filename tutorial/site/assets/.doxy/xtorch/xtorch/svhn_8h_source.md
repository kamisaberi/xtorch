

# File svhn.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**image-classification**](dir_9d21d6f83a70094db43fe94b096ae893.md) **>** [**svhn.h**](svhn_8h.md)

[Go to the documentation of this file](svhn_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class SVHN : BaseDataset {
        /*
        """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset where the data is stored.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
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
        explicit SVHN(const std::string &root);
        SVHN(const std::string &root, DataMode mode);
        SVHN(const std::string &root, DataMode mode , bool download);
        SVHN(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private:
        std::map<std::string, std::tuple<fs::path, fs::path, std::string>> resources = {
                {"train", {
                                  fs::path("http://ufldl.stanford.edu/housenumbers/train_32x32.mat"),
                                  fs::path("train_32x32.mat"),
                                  "e26dedcc434d2e4c54c9b2d4a06d8373"}
                },
                {"test",  {
                                  fs::path("http://ufldl.stanford.edu/housenumbers/test_32x32.mat"),
                                  fs::path("test_32x32.mat"),
                                  "eb5a983be6a315427106f1b164d9cef3"}
                },
                {"extra", {
                                  fs::path("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"),
                                  fs::path("extra_32x32.mat"),
                                  "a93ce644f1a588dc4d68dda5feec44a7"
                          }
                }
        };
        void load_data();

        void check_resources();

    };
}
```


