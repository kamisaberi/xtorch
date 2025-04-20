

# File flying-chairs.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**optical-flow**](dir_c5272f5a689662c2c5c28882f7ac0097.md) **>** [**flying-chairs.h**](flying-chairs_8h.md)

[Go to the documentation of this file](flying-chairs_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class FlyingChairs : public BaseDataset {
        /*
        """`FlyingChairs <https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs>`_ Dataset for optical flow.
        You will also need to download the FlyingChairs_train_val.txt file from the dataset page.
        The dataset is expected to have the following structure: ::

            root
                FlyingChairs
                    data
                        00001_flow.flo
                        00001_img1.ppm
                        00001_img2.ppm
                        ...
                    FlyingChairs_train_val.txt
        Args:
            root (str or ``pathlib.Path``): Root directory of the FlyingChairs Dataset.
            split (string, optional): The dataset split, either "train" (default) or "val"
            transforms (callable, optional): A function/transform that takes in
                ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
                ``valid_flow_mask`` is expected for consistency with other datasets which
                return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
        """
        */
    public :
        explicit FlyingChairs(const std::string &root);
        FlyingChairs(const std::string &root, DataMode mode);
        FlyingChairs(const std::string &root, DataMode mode , bool download);
        FlyingChairs(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };

}
```


