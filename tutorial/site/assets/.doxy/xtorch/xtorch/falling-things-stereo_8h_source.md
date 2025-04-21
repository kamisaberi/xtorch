

# File falling-things-stereo.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**stereo-matching-depth-estimation**](dir_e353cfd6010331702b3559c9641f7f23.md) **>** [**falling-things-stereo.h**](falling-things-stereo_8h.md)

[Go to the documentation of this file](falling-things-stereo_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class FallingThingsStereo : public BaseDataset {
        /*
        `FallingThings <https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation>`_ dataset.
        The dataset is expected to have the following structure: ::
        root
            FallingThings
                single
                    dir1
                        scene1
                            _object_settings.json
                            _camera_settings.json
                            image1.left.depth.png
                            image1.right.depth.png
                            image1.left.jpg
                            image1.right.jpg
                            image2.left.depth.png
                            image2.right.depth.png
                            image2.left.jpg
                            image2.right
                            ...
                        scene2
                    ...
                mixed
                    scene1
                        _object_settings.json
                        _camera_settings.json
                        image1.left.depth.png
                        image1.right.depth.png
                        image1.left.jpg
                        image1.right.jpg
                        image2.left.depth.png
                        image2.right.depth.png
                        image2.left.jpg
                        image2.right
                        ...
                    scene2
                    ...
         */
    public :
        explicit  FallingThingsStereo(const std::string &root);
        FallingThingsStereo(const std::string &root, DataMode mode);
        FallingThingsStereo(const std::string &root, DataMode mode , bool download);
        FallingThingsStereo(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        fs::path  url ="https://drive.google.com/open?id=1y4h9T6D9rf6dAmsRwEtfzJdcghCnI_01";
        fs::path dataset_file_name = "fat.zip";
        std::string dataset_file_md5 = "????????????????????????????????";
        fs::path dataset_folder_name = "FallingThings";
        void load_data();
        void check_resources();
    };
}
```


