

# File image-folder.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**general**](dir_3e490c73b2bbc01f3b90ef3b6e284c64.md) **>** [**image-folder.h**](image-folder_8h.md)

[Go to the documentation of this file](image-folder_8h.md)


```C++
#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    enum class LabelsType {
        BY_FOLDER = 0,
        BY_NAME = 1,
    };


    class ImageFolder : public BaseDataset {
    public :

        ImageFolder(const std::string &root);
        ImageFolder(const std::string &root,bool load_sub_folders);
        ImageFolder(const std::string &root,bool load_sub_folders, DataMode mode);
        ImageFolder(const std::string &root,bool load_sub_folders, DataMode mode , LabelsType label_type);
        ImageFolder(const std::string &root,bool load_sub_folders, DataMode mode , LabelsType label_type, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        LabelsType label_type = LabelsType::BY_FOLDER;
        bool load_sub_folders = false;
        void load_data();
    };
}
```


