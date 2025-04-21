

# File fgvc-aircraft.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**image-classification**](dir_9d21d6f83a70094db43fe94b096ae893.md) **>** [**fgvc-aircraft.h**](fgvc-aircraft_8h.md)

[Go to the documentation of this file](fgvc-aircraft_8h.md)


```C++
#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class FGVCAircraft :public BaseDataset {
    public :
        explicit FGVCAircraft(const std::string &root);
        FGVCAircraft(const std::string &root, DataMode mode);
        FGVCAircraft(const std::string &root, DataMode mode , bool download);
        FGVCAircraft(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        fs::path url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz";
        std::string dataset_file_md5 = "85eeb15f3717b99a5da872d97d918f87";
        std::string dataset_file_name = "fgvc-aircraft-2013b.tar.gz";
        fs::path dataset_folder_name = "fgvc-aircraft-2013b";

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
```


