

# File euro-sat.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**image-classification**](dir_9d21d6f83a70094db43fe94b096ae893.md) **>** [**euro-sat.h**](euro-sat_8h.md)

[Go to the documentation of this file](euro-sat_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class EuroSAT :public BaseDataset {
    public :
        explicit EuroSAT(const std::string &root);
        EuroSAT(const std::string &root, DataMode mode);
        EuroSAT(const std::string &root, DataMode mode , bool download);
        EuroSAT(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private:
        std::string url =
                "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip";
        fs::path archive_file_name = "EuroSAT.zip";
        std::string archive_file_md5 = "c8fa014336c82ac7804f0398fcb19387";
        fs::path dataset_folder_name = "euro-sat";

        void load_data();

        void check_resources();
    };
}
```


