

# File photo-tour.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**patch-matching-descriptor-learning**](dir_16503b39329efc9391e88fd543051250.md) **>** [**photo-tour.h**](photo-tour_8h.md)

[Go to the documentation of this file](photo-tour_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class PhotoTour : BaseDataset {
        /*
        """`Multi-view Stereo Correspondence <http://matthewalunbrown.com/patchdata/patchdata.html>`_ Dataset.

    .. note::

        We only provide the newer version of the dataset, since the authors state that it

            is more suitable for training descriptors based on difference of Gaussian, or Harris corners, as the
            patches are centred on real interest point detections, rather than being projections of 3D points as is the
            case in the old dataset.

        The original dataset is available under http://phototour.cs.washington.edu/patches/default.htm.


    Args:
        root (str or ``pathlib.Path``): Root directory where images are.
        name (string): Name of the dataset to load.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

         */
    public :
        explicit PhotoTour(const std::string &root);
        PhotoTour(const std::string &root, DataMode mode);
        PhotoTour(const std::string &root, DataMode mode , bool download);
        PhotoTour(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        std::map<std::string, std::tuple<fs::path, fs::path, std::string> > resources = {
            {
                "notredame_harris", {
                    fs::path("http://matthewalunbrown.com/patchdata/notredame_harris.zip"), fs::path(
                        "notredame_harris.zip"),
                    "69f8c90f78e171349abdf0307afefe4d"
                }
            },
            {
                "yosemite_harris", {
                    fs::path("http://matthewalunbrown.com/patchdata/yosemite_harris.zip"), fs::path(
                        "yosemite_harris.zip"),
                    "a73253d1c6fbd3ba2613c45065c00d46"
                }
            },
            {
                "liberty_harris", {
                    fs::path("http://matthewalunbrown.com/patchdata/liberty_harris.zip"), fs::path(
                        "liberty_harris.zip"),
                    "c731fcfb3abb4091110d0ae8c7ba182c"
                }
            },
            {
                "notredame", {
                    fs::path("http://icvl.ee.ic.ac.uk/vbalnt/notredame.zip"), fs::path(
                        "notredame.zip"),
                    "509eda8535847b8c0a90bbb210c83484"
                }
            },
            {
                "yosemite", {
                    fs::path("http://icvl.ee.ic.ac.uk/vbalnt/yosemite.zip"), fs::path(
                        "yosemite.zip"),
                    "533b2e8eb7ede31be40abc317b2fd4f0"
                }
            },
            {
                "liberty", {
                    fs::path("http://icvl.ee.ic.ac.uk/vbalnt/liberty.zip"), fs::path(
                        "liberty.zip"),
                    "fdd9152f138ea5ef2091746689176414"
                }
            },


        };

        fs::path dataset_folder_name = "photo-tour";
        void load_data();

        void check_resources();
    };
}
```


